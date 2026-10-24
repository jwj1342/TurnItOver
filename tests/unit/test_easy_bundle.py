import json
import hashlib
import shutil
from pathlib import Path

import pytest

from scripts.validate_easy_bundle import validate_bundle, write_manifest


ROOT = Path(__file__).parents[2] / "benchmark" / "easy"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def _refresh(root):
    split_path = root / "splits.json"
    split = json.loads(split_path.read_text())
    split["source"]["sha256"] = hashlib.sha256((root / "private/gold.jsonl").read_bytes()).hexdigest()
    _write_json(split_path, split)
    write_manifest(root)


def test_easy_bundle_integrity_and_counts():
    report = validate_bundle(ROOT)
    assert report["cases"] == 275
    assert report["objects"] == 127
    assert (report["train_objects"], report["train_cases"]) == (117, 251)
    assert (report["validation_objects"], report["validation_cases"]) == (10, 24)
    assert (report["imported_cases"], report["imported_objects"]) == (150, 75)
    assert report["media_files"] == 275 * 6
    assert report["independent_validation_review"]["cases"] == 24
    assert report["independent_validation_review"]["media_files"] == 144


def test_easy_bundle_rejects_imported_training_eligibility_before_review(tmp_path):
    copied_root = tmp_path / "easy"
    shutil.copytree(ROOT, copied_root)
    gold_path = copied_root / "private/gold.jsonl"
    gold_rows = [json.loads(line) for line in gold_path.read_text().splitlines()]
    imported = next(row for row in gold_rows if row.get("origin") == "medium-v1-reclassified-obvious")
    imported["training_eligibility"] = "eligible_individually_reviewed"
    _write_jsonl(gold_path, gold_rows)
    _refresh(copied_root)

    with pytest.raises(ValueError, match="not review-gated"):
        validate_bundle(copied_root)


@pytest.mark.parametrize("kind,verdict,accepted", [
    ("clean", "pass", True),
    ("clean", "fail", False),
    ("direction_flip", "fail", True),
    ("direction_flip", "pass", False),
])
def test_easy_bundle_accepts_only_matching_per_case_review(tmp_path, kind, verdict, accepted):
    root = tmp_path / "easy"
    shutil.copytree(ROOT, root)
    gold_path = root / "private/gold.jsonl"
    gold = [json.loads(line) for line in gold_path.read_text().splitlines()]
    imported = next(row for row in gold if row.get("origin") == "medium-v1-reclassified-obvious" and row["kind"] == kind)
    case_id = imported["case_id"]
    imported["training_eligibility"] = "eligible_individually_reviewed"
    _write_jsonl(gold_path, gold)
    for filename in ("human_review.jsonl", "review_manifest.jsonl"):
        path = root / filename
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        row = next(row for row in rows if row["case_id"] == case_id)
        row["status"] = "human_review_complete"
        if filename == "human_review.jsonl":
            row.update(reviewer="test-reviewer", reviewed_at="2026-10-24", review_mode="independent_per_case_blind_review")
            row["human_judgment"]["verdict"] = verdict
        _write_jsonl(path, rows)
    _refresh(root)
    if accepted:
        assert validate_bundle(root)["human_review_status"]["human_review_complete"] == 33
    else:
        with pytest.raises(ValueError, match="not review-gated"):
            validate_bundle(root)


def test_easy_bundle_rejects_dangling_program_reference(tmp_path):
    root = tmp_path / "easy"
    shutil.copytree(ROOT, root)
    path = root / "private/gold.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    next(row for row in rows if row.get("program"))["program"] = "missing/program.json"
    _write_jsonl(path, rows)
    _refresh(root)
    with pytest.raises(ValueError, match="program archive reference"):
        validate_bundle(root)


def test_easy_bundle_rejects_stale_current_gold_hash(tmp_path):
    root = tmp_path / "easy"
    shutil.copytree(ROOT, root)
    path = root / "splits.json"
    split = json.loads(path.read_text())
    split["source"]["sha256"] = "0" * 64
    _write_json(path, split)
    write_manifest(root)
    with pytest.raises(ValueError, match="current relative Gold"):
        validate_bundle(root)


def test_easy_bundle_rejects_original_split_reassignment(tmp_path):
    root = tmp_path / "easy"
    shutil.copytree(ROOT, root)
    path = root / "splits.json"
    split = json.loads(path.read_text())
    entry = split["objects"][0]
    entry["split"] = "validation" if entry["split"] == "train" else "train"
    _write_json(path, split)
    write_manifest(root)
    with pytest.raises(ValueError, match="object split assignment changed"):
        validate_bundle(root)


def test_easy_bundle_rejects_symlinked_media(tmp_path):
    root = tmp_path / "easy"
    shutil.copytree(ROOT, root)
    image = next((root / "inputs").glob("*.png"))
    original = ROOT / "inputs" / image.name
    image.unlink()
    image.symlink_to(original)
    with pytest.raises(ValueError, match="symbolic links"):
        validate_bundle(root)


def test_easy_bundle_rejects_archive_index_missing_case(tmp_path):
    root = tmp_path / "easy"
    shutil.copytree(ROOT, root)
    path = root / "archive/program_members.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _write_jsonl(path, rows[1:])
    write_manifest(root)
    with pytest.raises(ValueError, match="member index does not cover"):
        validate_bundle(root)


@pytest.mark.parametrize("field,value,error", [
    ("source_sha256", "0" * 64, "source hash differs"),
    ("source_path", "inputs/incorrect.png", "source identity differs"),
])
def test_easy_bundle_binds_blind_media_to_source(tmp_path, field, value, error):
    root = tmp_path / "easy"
    shutil.copytree(ROOT, root)
    path = root / "validation_review/media_provenance.json"
    rows = json.loads(path.read_text())
    rows[0][field] = value
    _write_json(path, rows)
    write_manifest(root)
    with pytest.raises(ValueError, match=error):
        validate_bundle(root)
