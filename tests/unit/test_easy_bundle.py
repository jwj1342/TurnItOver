"""Pure bundle checks use only deterministic synthetic data in tmp_path."""

import json

import pytest

from scripts.easy_bundle_contract import validate_blind_review_sources, validate_imported_review
from scripts.easy_bundle_provenance import validate_provenance
from scripts.validate_easy_bundle import _validate_manifest, validate_bundle
from tests.unit.easy_bundle_fixture import build_bundle, read_jsonl, refresh, write_json, write_jsonl


@pytest.fixture
def bundle(tmp_path):
    return build_bundle(tmp_path / "easy")


def _validate_provenance(root):
    gold = {row["case_id"]: row for row in read_jsonl(root / "private/gold.jsonl")}
    validate_provenance(root, gold, json.loads((root / "splits.json").read_text()))


def test_synthetic_bundle_manifest_and_provenance(bundle):
    files = _validate_manifest(bundle)
    assert len([name for name in files if name.endswith(".png")]) == 24
    _validate_provenance(bundle)
    reviews = {row["case_id"]: row for row in read_jsonl(bundle / "review_manifest.jsonl")}
    validate_blind_review_sources(bundle, reviews)


def test_production_validator_keeps_release_count_contract(bundle):
    with pytest.raises(ValueError, match="expected 275 Easy cases, found 4"):
        validate_bundle(bundle)


def test_manifest_rejects_changed_bytes(bundle):
    (bundle / "inputs/original-clean.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate_manifest(bundle)


def test_manifest_rejects_unlisted_file(bundle):
    (bundle / "unexpected.txt").write_text("not listed")
    with pytest.raises(ValueError, match="unlisted=.*unexpected"):
        _validate_manifest(bundle)


def test_manifest_rejects_missing_file(bundle):
    (bundle / "inputs/original-clean.png").unlink()
    with pytest.raises(ValueError, match="manifest file is missing"):
        _validate_manifest(bundle)


def test_manifest_rejects_symlinked_media(bundle, tmp_path):
    original = tmp_path / "outside.png"
    original.write_bytes(b"external")
    image = bundle / "inputs/original-clean.png"
    image.unlink()
    image.symlink_to(original)
    with pytest.raises(ValueError, match="symbolic links"):
        _validate_manifest(bundle)


def test_manifest_rejects_parent_traversal(bundle):
    path = bundle / "bundle_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside.txt"] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError, match="unsafe bundle path"):
        _validate_manifest(bundle)


def test_imported_training_eligibility_requires_per_case_review(bundle):
    gold = read_jsonl(bundle / "private/gold.jsonl")[2]
    gold["training_eligibility"] = "eligible_individually_reviewed"
    human = read_jsonl(bundle / "human_review.jsonl")[2]
    with pytest.raises(ValueError, match="not review-gated"):
        validate_imported_review(gold, human)


@pytest.mark.parametrize("kind,verdict,accepted", [
    ("clean", "pass", True), ("clean", "fail", False),
    ("direction_flip", "fail", True), ("direction_flip", "pass", False),
])
def test_only_matching_per_case_review_qualifies(bundle, kind, verdict, accepted):
    gold = next(row for row in read_jsonl(bundle / "private/gold.jsonl") if row.get("program") and row["kind"] == kind)
    gold["training_eligibility"] = "eligible_individually_reviewed"
    human = dict(case_id=gold["case_id"], status="human_review_complete", reviewer="synthetic-reviewer",
                 reviewed_at="2026-10-24", review_mode="independent_per_case_blind_review", human_judgment={"verdict": verdict})
    if accepted:
        validate_imported_review(gold, human)
    else:
        with pytest.raises(ValueError, match="not review-gated"):
            validate_imported_review(gold, human)


def test_provenance_rejects_dangling_program_reference(bundle):
    path = bundle / "private/gold.jsonl"
    rows = read_jsonl(path)
    rows[2]["program"] = "missing/program.json"
    write_jsonl(path, rows)
    refresh(bundle)
    with pytest.raises(ValueError, match="program archive reference"):
        _validate_provenance(bundle)


def test_provenance_rejects_stale_current_gold_hash(bundle):
    path = bundle / "splits.json"
    split = json.loads(path.read_text())
    split["source"]["sha256"] = "0" * 64
    write_json(path, split)
    with pytest.raises(ValueError, match="current relative Gold"):
        _validate_provenance(bundle)


def test_provenance_rejects_original_split_reassignment(bundle):
    path = bundle / "splits.json"
    split = json.loads(path.read_text())
    split["objects"][0]["split"] = "train"
    write_json(path, split)
    with pytest.raises(ValueError, match="object split assignment changed"):
        _validate_provenance(bundle)


def test_provenance_rejects_archive_index_missing_case(bundle):
    path = bundle / "archive/program_members.jsonl"
    write_jsonl(path, read_jsonl(path)[1:])
    with pytest.raises(ValueError, match="member index does not cover"):
        _validate_provenance(bundle)


def test_provenance_rejects_unverified_custody_copy(bundle):
    path = bundle / "archive/registry.json"
    registry = json.loads(path.read_text())
    registry["archives"][0]["custody_copies"][0]["verified_sha256"] = "0" * 64
    write_json(path, registry)
    with pytest.raises(ValueError, match="custody verification"):
        _validate_provenance(bundle)


def test_collection_inclusion_does_not_claim_independent_review(bundle):
    path = bundle / "collection_approval.json"
    approval = json.loads(path.read_text())
    approval["independent_per_case_review"] = True
    write_json(path, approval)
    with pytest.raises(ValueError, match="inclusion approval"):
        _validate_provenance(bundle)


@pytest.mark.parametrize("field,value,error", [
    ("source_sha256", "0" * 64, "source hash differs"),
    ("source_path", "inputs/incorrect.png", "source identity differs"),
])
def test_blind_media_is_bound_to_source(bundle, field, value, error):
    path = bundle / "validation_review/media_provenance.json"
    rows = json.loads(path.read_text())
    rows[0][field] = value
    write_json(path, rows)
    reviews = {row["case_id"]: row for row in read_jsonl(bundle / "review_manifest.jsonl")}
    with pytest.raises(ValueError, match=error):
        validate_blind_review_sources(bundle, reviews)
