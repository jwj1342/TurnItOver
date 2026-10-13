#!/usr/bin/env python3
"""Validate and summarize the formal PartNet-Mobility Train/Validation increment."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(asset_root: Path, benchmark_root: Path, test_plan: Path) -> dict:
    catalog = _json(asset_root / "catalog.json")
    splits = _json(asset_root / "splits.json")
    conversion = _json(asset_root / "conversion.json")
    prepared = benchmark_root / "prepared"
    episode_root = benchmark_root / "oracle-first-failure-v3"
    raw_manifest = _json(benchmark_root / "raw/shard-0000.manifest.json")
    prepared_manifest = _json(prepared / "manifest.json")
    episode_manifest = _json(episode_root / "manifest.json")
    test = _json(test_plan)

    if catalog.get("role") != "formal_train_validation" or not catalog.get("formal_training"):
        raise ValueError("Catalog is not marked as formal Train/Validation data")
    groups = splits["groups"]
    if set(groups.values()) - {"train", "validation"}:
        raise ValueError("Unexpected split name")
    entries = catalog["assets"]
    if len(entries) != len(groups) or len({row["split_group"] for row in entries}) != len(entries):
        raise ValueError("Every asset must have one independent split group")

    test_groups = {str(row.get("split_group", row["id"])) for row in test["objects"]}
    if test_groups & set(groups):
        raise ValueError("Train/Validation split groups overlap frozen Test")

    specs = {}
    archive_hashes = set()
    source_ids = set()
    annotation_ids = set()
    model_ids = set()
    for row in entries:
        spec_path = asset_root / row["spec"]
        if (_sha256(spec_path) != row["spec_sha256"] or not row.get("formal_training")
                or row.get("dataset") != "sapien-sim/PartNetMobility"
                or row.get("dataset_revision") != conversion["dataset_revision"]):
            raise ValueError(f"Spec hash/formal flag failed: {row['spec']}")
        spec = _json(spec_path)
        specs[spec["asset_id"]] = spec
        archive_hashes.add(row["archive_sha256"])
        source_ids.add(row["source_identity"])
        annotation_ids.add(row["annotation_id"])
        model_ids.add(row["model_id"])
    if any(len(values) != len(entries) for values in
           (archive_hashes, source_ids, annotation_ids, model_ids)):
        raise ValueError("Upstream identity or archive hash is not unique")

    gold_rows: dict[str, list[dict]] = {}
    for split in ("train", "validation"):
        rows = _jsonl(prepared / f"{split}.gold.jsonl")
        gold_rows[split] = rows
        for row in rows:
            provenance = row.get("provenance", {})
            if (groups[row["split_group"]] != split or not provenance.get("formal_training")
                    or provenance.get("source_identity") != row["split_group"]):
                raise ValueError(f"Prepared provenance/split failed: {row['sample_id']}")
    accepted = [row for rows in gold_rows.values() for row in rows]
    if len(accepted) != raw_manifest["n_accepted"]:
        raise ValueError("Prepared accepted count differs from raw manifest")

    variants, defects = Counter(), Counter()
    for row in accepted:
        findings = row["target"]["findings"]
        variants["clean" if not findings else "single" if len(findings) == 1 else "combined"] += 1
        defects.update(finding["defect_id"] for finding in findings)

    episode_rows, action_rows = {}, {}
    for split in ("train", "validation"):
        episode_rows[split] = len(_jsonl(episode_root / f"{split}.episodes.jsonl"))
        action_rows[split] = len(_jsonl(episode_root / f"{split}.actions.jsonl"))
    if episode_rows != episode_manifest["episode_examples"]:
        raise ValueError("Episode row counts differ from manifest")
    if action_rows != episode_manifest["action_examples"]:
        raise ValueError("Action row counts differ from manifest")

    split_assets = Counter(groups[row["split_group"]] for row in entries)
    categories = {split: Counter() for split in ("train", "validation")}
    joints = {split: Counter() for split in ("train", "validation")}
    multi_joint = Counter()
    mixed_joint = Counter()
    for row in entries:
        spec = specs[Path(row["spec"]).stem]
        split = groups[row["split_group"]]
        categories[split][spec["category"]] += 1
        joint_types = Counter(joint["type"] for joint in spec["joints"])
        joints[split].update(joint_types)
        multi_joint[split] += len(spec["joints"]) > 1
        mixed_joint[split] += len(joint_types) > 1

    review_rows = [row for row in conversion["assets"] if row.get("motion_review")]
    passed_reviews = sum(row["motion_review"]["decision"] == "passed" for row in review_rows)
    structural_rejections = Counter()
    for row in conversion["assets"]:
        reason = row.get("rejection_reason")
        if not reason:
            continue
        if reason.startswith("Unsupported joint type or mimic"):
            reason = "unsupported joint type or mimic"
        structural_rejections[reason] += 1

    return {
        "version": 1, "status": "passed", "formal_training": True,
        "dataset_revision": conversion["dataset_revision"],
        "assets": {
            "total": len(entries), "by_split": dict(split_assets),
            "categories_total": len(set(categories["train"]) | set(categories["validation"])),
            "categories_by_split": {key: dict(sorted(value.items())) for key, value in categories.items()},
            "joints_by_split": {key: dict(value) for key, value in joints.items()},
            "multi_joint_by_split": dict(multi_joint),
            "mixed_joint_type_by_split": dict(mixed_joint),
        },
        "selection": {
            "downloaded": conversion["summary"]["selected"],
            "structural_candidates": conversion["summary"]["structural_candidates"],
            "motion_reviewed": len(review_rows), "motion_passed": passed_reviews,
            "motion_pass_rate": passed_reviews / len(review_rows),
            "structural_rejections": dict(structural_rejections),
            "motion_rejected": len(review_rows) - passed_reviews,
        },
        "supervision": {
            "generated": raw_manifest["n_ok"], "accepted": len(accepted),
            "quarantined": raw_manifest["n_quarantined"],
            "accepted_by_split": {key: len(value) for key, value in gold_rows.items()},
            "accepted_variants": dict(variants), "accepted_defects": dict(defects),
        },
        "episodes": {
            "samples": sum(episode_manifest["samples"].values()),
            "action_examples": action_rows, "episode_examples": episode_rows,
            "termination_reasons": episode_manifest["episode_termination_reasons"],
            "teacher_renders": episode_manifest["teacher_renders"],
            "exposure_recall": episode_manifest["exposure_recall"],
        },
        "isolation": {
            "unique_source_identities": len(source_ids),
            "unique_annotation_ids": len(annotation_ids), "unique_model_ids": len(model_ids),
            "unique_archive_hashes": len(archive_hashes),
            "train_validation_group_overlap": 0, "frozen_test_group_overlap": 0,
            "formal_test_plan_sha256": conversion["formal_test_plan_sha256"],
        },
        "quality": {
            "raw_status": raw_manifest["status"], "prepared_status": prepared_manifest["status"],
            "episode_status": episode_manifest["status"],
            "pixel_replay_validation": episode_manifest["replay_validation"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("data/assets/partnet-mobility-trainval-v1"))
    parser.add_argument("--benchmark", type=Path,
                        default=Path("data/benchmarks/partnet-mobility-formal-v1"))
    parser.add_argument("--test-plan", type=Path, default=Path("configs/uniphys-test.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/benchmarks/partnet-mobility-formal-v1/dataset-audit.json"))
    args = parser.parse_args()
    result = audit(args.assets.resolve(), args.benchmark.resolve(), args.test_plan.resolve())
    write_json_atomic(args.out.resolve(), result)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
