import hashlib
import json

import pytest

from turnitover.verifier.acquisition_scoring import score_acquisition


def _write_batch(root, manifest_sha, *, action="view:front"):
    root.mkdir()
    (root / "batch-manifest.json").write_text(json.dumps({
        "status": "complete", "score_eligible": True,
        "benchmark_manifest_sha256": manifest_sha,
        "config": {"budget": 1},
        "tasks": [{"id": "case-a"}, {"id": "case-b"}],
    }))
    for case_id, view in (("case-a", action), ("case-b", "view:front")):
        folder = root / case_id
        folder.mkdir()
        view_id = view.removeprefix("view:")
        (folder / "result.json").write_text(json.dumps({
            "status": "complete", "trajectory": [{"action": {"__type__": "RequestView", "view_id": view_id}}],
        }))


def _sidecar(path, manifest_sha):
    value = {
        "schema_version": 1, "benchmark_manifest_sha256": manifest_sha,
        "oracle": {"exposure_rate": {"1": 1.0}},
        "cases": [
            {"id": "case-a", "multi_joint": False,
             "prefixes": [{"actions": ["view:front"], "exposed": True}]},
            {"id": "case-b", "multi_joint": True,
             "prefixes": [{"actions": ["view:front"], "exposed": False}]},
        ],
    }
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scores_hash_bound_exposure_curves(tmp_path):
    root, sidecar, output = tmp_path / "run", tmp_path / "sidecar.json", tmp_path / "score.json"
    _write_batch(root, "frozen-manifest")
    sidecar_hash = _sidecar(sidecar, "frozen-manifest")

    report = score_acquisition({"fixed": root}, sidecar, output)

    assert report["exposure_sidecar_sha256"] == sidecar_hash
    assert report["runs"]["fixed"]["exposure_rate"] == .5
    assert report["runs"]["fixed"]["multi_joint_exposure_rate"] == 0.0
    assert report["oracle"] == {"1": 1.0}


def test_rejects_missing_prefix_instead_of_guessing_exposure(tmp_path):
    root, sidecar = tmp_path / "run", tmp_path / "sidecar.json"
    _write_batch(root, "frozen-manifest", action="view:side")
    _sidecar(sidecar, "frozen-manifest")

    with pytest.raises(ValueError, match="missing exposure record"):
        score_acquisition({"fixed": root}, sidecar, tmp_path / "score.json")
