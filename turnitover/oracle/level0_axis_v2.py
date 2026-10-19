"""Strict provenance audit and frozen source-object split for Level 0 v2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from .level0_axis import PROMPT, load_records, read_jsonl, sha256

SOURCE_COMMIT = "d5cbc085a4dbb7b573aa17f42ec67b322d0003b3"
# All five objects in the previous binary SFT and fixed-evidence replay.
OLD_LEVEL0_ASSETS = {"PM_10697", "PM_12036", "PM_102016", "PM_12594", "PM_5088"}


def _indexed(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    indexed = {row["case_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate case ID: {path.name}")
    return indexed


def prepare_v2(dataset: Path, out: Path, seed: int) -> None:
    """Validate before creating output; no result-dependent split selection."""
    manifest = json.loads((dataset / "manifest.json").read_text())
    actual_files = {str(p.relative_to(dataset)) for p in dataset.rglob("*") if p.is_file()}
    if actual_files != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Source manifest file coverage mismatch")
    for relative, expected in manifest["files"].items():
        path = dataset / relative
        if not path.resolve().is_relative_to(dataset.resolve()) or sha256(path) != expected:
            raise ValueError(f"Source file SHA mismatch: {relative}")
    reviews = _indexed(dataset / "human_review.jsonl")
    gold = _indexed(dataset / "private/gold.jsonl")
    targets = _indexed(dataset / "private/train_targets.jsonl")
    training = _indexed(dataset / "train_manifest.jsonl")
    review_manifest = _indexed(dataset / "review_manifest.jsonl")
    case_ids = set(training)
    if len(case_ids) != 56 or any(set(rows) != case_ids for rows in (reviews, gold, targets, review_manifest)):
        raise ValueError("Expected identical 56-case reviewed/eligible membership")
    validation = json.loads((dataset / "validation.json").read_text())
    if validation.get("acceptance") != "complete" or validation.get("pending_human_review") != 0:
        raise ValueError("Source acceptance incomplete")
    for case in sorted(case_ids):
        label = reviews[case]["human_judgment"]["verdict"]
        if (reviews[case]["status"] != "human_review_complete"
                or review_manifest[case]["status"] != "human_review_complete"
                or label not in {"pass", "fail"}
                or label != gold[case]["expected_status"]
                or label != targets[case]["expected_status"]
                or gold[case].get("training_eligibility") != "eligible"):
            raise ValueError(f"Review/Gold/target eligibility mismatch: {case}")
        if (gold[case]["kind"], label) not in {("clean", "pass"), ("joint_axis", "fail")}:
            raise ValueError(f"Unexpected task kind: {case}")
        row = training[case]
        if row["input_preview"] != f"inputs/{case}.png" or row["images"] != review_manifest[case]["images"]:
            raise ValueError(f"Input identity mismatch: {case}")
        with Image.open(dataset / row["input_preview"]) as image:
            if image.size != (542, 542):
                raise ValueError(f"Unexpected four-panel dimensions: {case}")
    objects = json.loads((dataset / "object_splits.json").read_text())
    if objects["unit"] != "independent_source_object" or len(objects["objects"]) != 28:
        raise ValueError("Expected 28 independent source objects")
    objects = objects["objects"]
    catalog = json.loads((dataset / manifest["source_catalog"]).read_text())["assets"]
    source = {row["source_identity"]: row for row in catalog}
    if len(source) != 28 or len(catalog) != 28:
        raise ValueError("Duplicate source identity")
    # These identifiers must each be independent, not merely different asset IDs.
    for field in ("asset_id", "split_group", "source_model_id", "source_archive_sha256"):
        if len({row[field] for row in objects}) != 28:
            raise ValueError(f"Duplicate object identity: {field}")
    for field in ("model_id", "archive_sha256", "original_geometry_fingerprint"):
        if len({row[field] for row in catalog}) != 28:
            raise ValueError(f"Duplicate source geometry identity: {field}")
    upstream_splits = json.loads((dataset / manifest["expanded_splits"]).read_text())["groups"]
    seen_cases = []
    prior = set(OLD_LEVEL0_ASSETS)
    inherited = json.loads((dataset / "private/inherited_review_provenance.json").read_text())
    prior.update(gold[case]["asset_id"] for case in inherited)
    for obj in objects:
        identity = obj["split_group"]
        origin = source[identity]
        if (obj["split"] != "train" or not obj["training_eligible"]
                or upstream_splits[identity] != "train"
                or origin["source_identity"] != identity
                or identity != f"PartNet-Mobility:{obj['asset_id'].removeprefix('PM_')}"
                or origin["model_id"] != obj["source_model_id"]
                or origin["archive_sha256"] != obj["source_archive_sha256"]
                or origin.get("upstream_structural_checks", {}).get("frozen_test_overlap", False)):
            raise ValueError(f"Source identity/eligibility mismatch: {identity}")
        reference = dataset / "private/source_catalog" / origin["spec"]
        if sha256(reference) != origin["spec_sha256"]:
            raise ValueError(f"Source spec SHA mismatch: {identity}")
        cases = obj["case_ids"]
        if (len(cases) != 2 or set(cases) - case_ids
                or {gold[case]["kind"] for case in cases} != {"clean", "joint_axis"}
                or any(gold[case]["asset_id"] != obj["asset_id"] for case in cases)):
            raise ValueError(f"Invalid clean/axis pair: {identity}")
        seen_cases.extend(cases)
        if origin.get("previous_motion_review") is not None:
            prior.add(obj["asset_id"])
    if len(seen_cases) != 56 or set(seen_cases) != case_ids:
        raise ValueError("Object pairing does not cover every case exactly once")
    # Hash ordering avoids dependency on Python's PRNG version and never uses labels/results.
    eligible = [obj for obj in objects if obj["asset_id"] not in prior]
    eligible.sort(key=lambda obj: hashlib.sha256(f"{seed}:{obj['split_group']}".encode()).hexdigest())
    if len(eligible) < 6:
        raise ValueError("Insufficient new objects for held-out split")
    heldout_assets = {obj["asset_id"] for obj in eligible[:6]}
    train = sorted(case for obj in objects if obj["asset_id"] not in heldout_assets for case in obj["case_ids"])
    heldout = sorted(case for obj in objects if obj["asset_id"] in heldout_assets for case in obj["case_ids"])
    records = load_records(dataset)
    if len(records) != 56:
        raise ValueError("Loader did not preserve manifest membership")
    split = {
        "source_commit": SOURCE_COMMIT, "seed": seed,
        "task": "four-panel motion comparison -> pass|fail", "prompt": PROMPT,
        "train_case_ids": train, "heldout_case_ids": heldout,
        "train_assets": 22, "heldout_assets": 6,
        "train_asset_ids": sorted(obj["asset_id"] for obj in objects if obj["asset_id"] not in heldout_assets),
        "heldout_asset_ids": sorted(heldout_assets),
        "selection_rule": "First six unseen source objects ordered by SHA256(seed:split_group); frozen before training",
        "prior_exposure_excluded_from_heldout": sorted(prior),
        "prior_exposure_definition": "Old binary/fixed-evidence objects, inherited v1 reviews, or any previous_motion_review",
        "asset_pair_crosses_split": False,
        "model_visible_fields": ["image", "prompt"],
        "source_manifest_prompt_used": False,
        "private_gold_used_only_for": "review equality, eligibility, clean/axis pairing, source identity and prior exposure audit",
    }
    audit = {
        "status": "passed", "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": sha256(dataset / "manifest.json"),
        "verified_manifest_file_count": len(manifest["files"]),
        "verified_file_sha256": manifest["files"],
        "input_sha256": {case: sha256(dataset / training[case]["input_preview"]) for case in sorted(case_ids)},
        "cases": 56, "asset_pairs": 28, "train_cases": 44, "heldout_cases": 12,
        "input_size": [542, 542], "native_tile_size": [256, 256],
        "human_gold_train_targets_agree": True, "source_identity_unique": True,
        "upstream_train_only": True, "new_heldout_objects": 6,
        "external_input_hashes": manifest["inputs"],
        "external_input_hash_scope": "Recorded upstream hashes; external absolute paths are not re-read",
    }
    out.mkdir(parents=True)
    (out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    audit["split_sha256"] = sha256(out / "split.json")
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
