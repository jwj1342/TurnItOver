"""Regression checks for audited v2 membership and source-object splitting."""
import json
from pathlib import Path

import pytest
from PIL import Image

from turnitover.oracle.level0_axis import PROMPT, load_records, sha256
from turnitover.oracle.level0_axis_v2 import prepare_v2


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _refresh_manifest(dataset):
    manifest = {
        "version": 2, "source_catalog": "private/source_catalog/catalog.json",
        "expanded_splits": "private/expanded_splits.json", "inputs": {},
        "files": {str(path.relative_to(dataset)): sha256(path) for path in dataset.rglob("*")
                  if path.is_file() and path.name != "manifest.json"},
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "source"
    (root / "inputs").mkdir(parents=True)
    (root / "private/source_catalog").mkdir(parents=True)
    reviews, gold, targets, training, review_manifest, objects, catalog = ([] for _ in range(7))
    for index in range(28):
        asset, identity = f"PM_{index}", f"PartNet-Mobility:{index}"
        cases = [f"case-{index}-{kind}" for kind in ("clean", "joint_axis")]
        spec = root / "private" / f"spec-{index}.json"
        spec.write_text(json.dumps({"asset_id": asset}))
        objects.append({"asset_id": asset, "split_group": identity, "split": "train",
                        "training_eligible": True, "case_ids": cases,
                        "source_model_id": f"model-{index}", "source_archive_sha256": f"archive-{index}"})
        catalog.append({"source_identity": identity, "model_id": f"model-{index}",
                        "archive_sha256": f"archive-{index}", "original_geometry_fingerprint": f"geometry-{index}",
                        "spec": f"../spec-{index}.json", "spec_sha256": sha256(spec),
                        "previous_motion_review": {"decision": "passed"} if index < 7 else None})
        for case, kind in zip(cases, ("clean", "joint_axis")):
            label = "pass" if kind == "clean" else "fail"
            Image.new("RGB", (542, 542)).save(root / "inputs" / f"{case}.png")
            reviews.append({"case_id": case, "status": "human_review_complete", "human_judgment": {"verdict": label}})
            gold.append({"case_id": case, "asset_id": asset, "kind": kind,
                         "expected_status": label, "training_eligibility": "eligible"})
            targets.append({"case_id": case, "expected_status": label})
            training.append({"case_id": case, "input_preview": f"inputs/{case}.png", "images": [], "prompt": "PRIVATE WRONG PROMPT"})
            review_manifest.append({"case_id": case, "images": [], "status": "human_review_complete"})
    for relative, rows in [("human_review.jsonl", reviews), ("private/gold.jsonl", gold),
                           ("private/train_targets.jsonl", targets), ("train_manifest.jsonl", training),
                           ("review_manifest.jsonl", review_manifest)]:
        _write_rows(root / relative, rows)
    for relative, content in [("object_splits.json", {"unit": "independent_source_object", "objects": objects}),
                              ("private/source_catalog/catalog.json", {"assets": catalog}),
                              ("private/expanded_splits.json", {"groups": {obj["split_group"]: "train" for obj in objects}}),
                              ("private/inherited_review_provenance.json", {}),
                              ("validation.json", {"acceptance": "complete", "pending_human_review": 0})]:
        (root / relative).write_text(json.dumps(content))
    _refresh_manifest(root)
    return root


def test_frozen_split_is_paired_new_and_deterministic(dataset, tmp_path):
    prepare_v2(dataset, tmp_path / "a", 20261017)
    prepare_v2(dataset, tmp_path / "b", 20261017)
    assert (tmp_path / "a/split.json").read_bytes() == (tmp_path / "b/split.json").read_bytes()
    split = json.loads((tmp_path / "a/split.json").read_text())
    assert len(split["train_case_ids"]) == 44
    assert len(split["heldout_case_ids"]) == 12
    assert not set(split["heldout_asset_ids"]) & {f"PM_{i}" for i in range(7)}
    assert not set(split["train_asset_ids"]) & set(split["heldout_asset_ids"])
    assert split["prompt"] == PROMPT
    assert set(load_records(dataset)[0]) == {"case_id", "image", "label"}


def test_rejects_image_tamper_before_output(dataset, tmp_path):
    (dataset / "inputs/case-0-clean.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA mismatch"):
        prepare_v2(dataset, tmp_path / "out", 20261017)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["label", "duplicate_identity", "ineligible", "pair_cross_object"])
def test_rejects_semantic_provenance_errors(dataset, tmp_path, change):
    if change == "label":
        path = dataset / "private/train_targets.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["expected_status"] = "fail"
        _write_rows(path, rows)
    else:
        path = dataset / "object_splits.json"
        content = json.loads(path.read_text())
        if change == "duplicate_identity":
            content["objects"][1]["source_model_id"] = content["objects"][0]["source_model_id"]
        elif change == "ineligible":
            content["objects"][0]["training_eligible"] = False
        else:
            content["objects"][0]["case_ids"][0] = "case-1-clean"
        path.write_text(json.dumps(content))
    _refresh_manifest(dataset)
    with pytest.raises(ValueError):
        prepare_v2(dataset, tmp_path / "out", 20261017)
    assert not (tmp_path / "out").exists()


def test_loader_manifest_filters_reviews_and_rejects_duplicates(dataset):
    path = dataset / "train_manifest.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    _write_rows(path, [row])
    assert [r["case_id"] for r in load_records(dataset)] == [row["case_id"]]
    _write_rows(path, [row, row])
    with pytest.raises(ValueError, match="manifest membership"):
        load_records(dataset)


def test_legacy_loader_without_manifest(tmp_path):
    (tmp_path / "inputs").mkdir()
    Image.new("RGB", (542, 542)).save(tmp_path / "inputs/opaque.png")
    _write_rows(tmp_path / "human_review.jsonl", [{"case_id": "opaque", "status": "human_review_complete",
                                                 "human_judgment": {"verdict": "pass"}}])
    assert load_records(tmp_path) == [{"case_id": "opaque", "image": tmp_path / "inputs/opaque.png", "label": "pass"}]
