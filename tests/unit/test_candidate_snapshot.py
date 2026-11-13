"""Native candidates retain variable inputs and independent private truth."""
import hashlib
import json
import shutil

import pytest

from turnitover.oracle.candidate_scoring import score_candidate_snapshot
from turnitover.oracle.candidate_snapshot import prepare_candidate_snapshot

LABEL = "kinematics.joint_axis"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _dataset(tmp_path):
    root = tmp_path / "dataset"
    files, cases = {}, []
    for uid, pairs, clean in (("a", 1, False), ("b", 5, True)):
        images = []
        for number in range(pairs):
            pair = {"state": str(number), "view_id": "front"}
            for role in ("reference_image", "candidate_image"):
                relative = f"assets/{uid}-{number}-{role}.png"
                content = f"{uid}-{number}".encode()
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                pair[role] = {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}
            images.append(pair)
        labels = [] if clean else [LABEL]
        gold_path = f"assets/{uid}.json"
        gold_sha = _write(root / gold_path, {"case_id": uid, "defect_ids": labels})
        case = {"id": uid, "split": "validation", "tier": "easy", "clean": clean,
                "labels": labels, "source_key": "source:" + uid,
                "human_review_status": "pending_human_review", "training_eligible": False,
                "formal_admitted": False, "observations": images,
                "provenance": {"retained": True}, "evidence_files": [{"path": "unopened-history.json"}],
                "assets": {"gold_file": {"path": gold_path, "sha256": gold_sha},
                           "clean_program": {"sha256": "same"},
                           "candidate_program": {"sha256": "same" if clean else "different"}}}
        relative = f"cases/validation/easy/{uid}.json"
        files[relative] = {"sha256": _write(root / relative, case)}
        cases.append({"id": uid, "path": relative})
    # A sealed, deliberately nonexistent case proves snapshot creation selects paths first.
    cases.append({"id": "sealed", "path": "cases/test/easy/sealed.json"})
    manifest = {"format": "tio-750-complete-candidate-custody-v1", "cases": cases, "files": files,
                "status": "complete_candidate_materials_pending_human_review",
                "human_review_status": "pending_human_review", "training_eligible_cases": 0,
                "formal_admitted_cases": 0}
    return root, _write(root / "manifest.private.json", manifest)


def _predictions(path, raw_a=None):
    values = [
        {"case_id": "a", "raw": raw_a if raw_a is not None else json.dumps({"verdict": {
            "status": "fail", "findings": [{"defect_id": LABEL, "evidence_steps": [0]}]}}),
         "input_tokens": 7, "usage": {"output_tokens": 3}},
        {"id": "b", "raw": json.dumps({"verdict": {"status": "pass", "findings": []}})},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in values))
    return path


def test_variable_pairs_private_truth_and_dataset_deletion(tmp_path):
    dataset, sha = _dataset(tmp_path)
    snapshot = tmp_path / "snapshot"
    manifest = prepare_candidate_snapshot(dataset, snapshot, manifest_sha256=sha)
    inputs = [json.loads(line) for line in (snapshot / "inputs.jsonl").read_text().splitlines()]
    assert [len(row["observations"]) for row in inputs] == [1, 5]
    assert [len(row["images"]) for row in inputs] == [2, 10]
    assert all(set(row) == {"id", "images", "observations"} for row in inputs)
    assert len(list((snapshot / "images").iterdir())) == 6  # Deduplicate identical paired bytes.
    assert manifest["training_eligible_cases"] == 0
    assert json.loads((snapshot / "cases/validation/easy/a.json").read_text())["provenance"] == {"retained": True}
    (dataset / "manifest.private.json").write_text("changed")
    predictions = _predictions(tmp_path / "predictions.jsonl")
    first = score_candidate_snapshot(snapshot, predictions, tmp_path / "score1")
    shutil.rmtree(dataset)
    second = score_candidate_snapshot(snapshot, predictions, tmp_path / "score2")
    assert first == second
    assert first["summary"]["defect_macro_f1"] == 1 / 29
    assert first["summary"]["supported_label_macro_f1"] == 1
    assert first["summary"]["localization"]["value"] is None
    assert first["summary"]["exposure"]["value"] is None
    assert first["summary"]["false_pass_rate"] == 0
    scored = json.loads((tmp_path / "score1/cases.private.jsonl").read_text().splitlines()[0])
    assert scored["input_tokens"] == 7
    assert scored["usage"] == {"output_tokens": 3}
    assert scored["training_eligible"] is False


@pytest.mark.parametrize("raw", ["not json", '{"verdict":{"status":"fail","findings":[{"defect_id":"bogus"}]}}',
                                 '{"verdict":{"status":"pass","findings":[],"status":"fail"}}'])
def test_invalid_prediction_retains_false_negatives(tmp_path, raw):
    dataset, sha = _dataset(tmp_path)
    snapshot = tmp_path / "snapshot"
    prepare_candidate_snapshot(dataset, snapshot, manifest_sha256=sha)
    report = score_candidate_snapshot(snapshot, _predictions(tmp_path / "p.jsonl", raw), tmp_path / "score")
    assert report["summary"]["invalid_predictions"] == 1
    assert report["summary"]["per_label"][LABEL]["fn"] == 1
    assert report["summary"]["defect_macro_f1"] == 0


def test_coverage_and_snapshot_corruption(tmp_path):
    dataset, sha = _dataset(tmp_path)
    snapshot = tmp_path / "snapshot"
    prepare_candidate_snapshot(dataset, snapshot, manifest_sha256=sha)
    predictions = _predictions(tmp_path / "p.jsonl")
    predictions.write_text(predictions.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="cover every"):
        score_candidate_snapshot(snapshot, predictions, tmp_path / "missing")
    _predictions(predictions)
    image = next((snapshot / "images").iterdir())
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        score_candidate_snapshot(snapshot, predictions, tmp_path / "corrupt")


def test_manifest_and_source_hash_fail_closed(tmp_path):
    dataset, sha = _dataset(tmp_path)
    with pytest.raises(ValueError, match="manifest hash"):
        prepare_candidate_snapshot(dataset, tmp_path / "bad", manifest_sha256="0" * 64)
    case = dataset / "cases/validation/easy/a.json"
    case.write_text("changed")
    with pytest.raises(ValueError, match="Hash mismatch"):
        prepare_candidate_snapshot(dataset, tmp_path / "changed", manifest_sha256=sha)


@pytest.mark.parametrize("finding, valid", [
    ({"defect_id": LABEL, "evidence_observations": [1]}, True),
    ({"defect_id": LABEL, "evidence_observations": [1], "evidence_steps": [0]}, True),
    ({"defect_id": LABEL, "evidence_observations": [1], "evidence_steps": []}, False),
])
def test_one_based_native_citations(tmp_path, finding, valid):
    dataset, sha = _dataset(tmp_path)
    snapshot = tmp_path / "snapshot"
    prepare_candidate_snapshot(dataset, snapshot, manifest_sha256=sha)
    raw = json.dumps({"verdict": {"status": "fail", "findings": [finding]}})
    report = score_candidate_snapshot(snapshot, _predictions(tmp_path / "p.jsonl", raw), tmp_path / "score")
    assert report["summary"]["invalid_predictions"] == int(not valid)
    if valid:
        row = json.loads((tmp_path / "score/cases.private.jsonl").read_text().splitlines()[0])
        assert row["parsed"]["findings"][0]["evidence_steps"] == [0]
        assert row["raw"] == raw
