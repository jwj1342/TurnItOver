import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image

from turnitover.oracle.release_sft_export import export_releases
from turnitover.oracle.sft_data import load_training_examples


def _png(path: Path, color: str) -> bytes:
    Image.new("RGB", (4, 4), color).save(path)
    return path.read_bytes()


def test_release_export_loads_all_sft_objectives(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    (release / "release-manifest.json").write_text(json.dumps({"tag": "fixture-medium"}))
    reference = _png(tmp_path / "reference.png", "red")
    candidate = _png(tmp_path / "candidate.png", "blue")
    source = {"source_dataset": "sapien-sim/PartNetMobility", "source_asset_id": "42",
              "source_identity": "PartNet-Mobility:42", "original_spec_sha256": "a" * 64}
    gold = {"case_id": "ms-fixture", "asset_id": "PM_42", "kind": "direction_flip",
            "defect_id": "kinematics.direction_flip", "joint_id": "joint_0", "parts": ["link_0"],
            "parameters": {"joint": "joint_0"}, "reference_sha": "b" * 16,
            "candidate_sha": "c" * 16, "source": source}
    public = {"case_id": "ms-fixture", "view": "front", "images": []}
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "PM_42.json").write_text(json.dumps({
        "parts": [{"id": "base"}, {"id": "link_0"}, {"id": "link_1"}],
        "joints": [{"id": "joint_0"}, {"id": "joint_1"}],
    }))
    source["original_spec_sha256"] = hashlib.sha256((assets / "PM_42.json").read_bytes()).hexdigest()
    with zipfile.ZipFile(release / "fixture-data.zip", "w") as archive:
        archive.writestr("medium/private/gold.jsonl", json.dumps(gold) + "\n")
        archive.writestr("medium/review_manifest.jsonl", json.dumps(public) + "\n")
        for kind in ("reference", "candidate"):
            for state in ("rest", "mid", "limit"):
                archive.writestr(f"medium/native/ms-fixture/{kind}_{state}.png",
                                 reference if kind == "reference" else candidate)
    output = tmp_path / "export"
    manifest = export_releases([release], output, "fixture-v1", [assets])
    split = manifest["groups"]["sapien-sim/PartNetMobility:42"]
    for task in ("diagnosis", "stopping"):
        for supervision in ("ordinary", "evidence"):
            rows = list(load_training_examples(output, split, task, supervision))
            assert len(rows) == 1
            assert len(rows[0][1]) == 6
    record = json.loads((output / f"{split}.episodes.jsonl").read_text())
    assert record["image_sha256"] == [hashlib.sha256(reference).hexdigest()] * 3 + [
        hashlib.sha256(candidate).hexdigest()] * 3
    assert record["difficulty"] == "medium"
    payload = json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])
    assert payload["context"]["parts"] == ["base", "link_0", "link_1"]
    assert payload["context"]["joints"] == ["joint_0", "joint_1"]
    finding = json.loads(record["messages"][1]["content"])["verdict"]["findings"][0]
    assert finding["evidence_steps"] == [2, 3, 4]
