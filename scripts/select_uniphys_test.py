#!/usr/bin/env python3
"""Freeze unseen UniPhys objects for the formal test benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.select_uniphys_trainval import DATASET, LICENSE, OBJECT_RE, _convertible, _eligible
from turnitover.core.jsonio import write_json_atomic


DEV_RANGE = {f"UPB_{index:08d}" for index in range(20)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _metadata(folder: Path) -> dict:
    record = json.loads((folder / "meta_data.json").read_text(encoding="utf-8"))
    source, obj = record["source"], record["object"]
    return {
        "category": obj["category"],
        "name_key": _normalized_name(obj["object_name"]),
        "source_key": f"{source['dataset']}:{source['original_id']}",
        "geometry_sha256": _sha256(folder / "full_model/model.obj"),
    }


def _split_ids(path: Path, expected: set[str]) -> set[str]:
    record = json.loads(path.read_text(encoding="utf-8"))
    groups = record.get("groups")
    if (record.get("version") != 1 or not isinstance(groups, dict) or not groups
            or not set(groups.values()) <= expected):
        raise ValueError(f"Invalid split manifest: {path}")
    ids = {group.rsplit(":", 1)[-1] for group in groups}
    if any(not OBJECT_RE.fullmatch(object_id) for object_id in ids):
        raise ValueError(f"Invalid object ID in split manifest: {path}")
    return ids


def select(root: Path, previous_test_splits: Path, trainval_splits: Path,
           trainval_plan: Path, output: Path, *, objects: int = 20,
           seed: int = 20261011) -> dict:
    if output.exists():
        raise ValueError(f"Selection output already exists: {output}")
    if objects < 1:
        raise ValueError("Test object count must be positive")
    acquisition = json.loads((root / "download-manifest.json").read_text(encoding="utf-8"))
    revision = acquisition.get("revision")
    if (acquisition.get("dataset") != DATASET or acquisition.get("license") != LICENSE
            or not isinstance(revision, str) or len(revision) != 40):
        raise ValueError("Expected a pinned UniPhys-Bench Part 2 acquisition")
    previous_test_ids = _split_ids(previous_test_splits, {"test"})
    trainval_ids = _split_ids(trainval_splits, {"train", "validation"})
    plan = json.loads(trainval_plan.read_text(encoding="utf-8"))
    if ({row.get("id") for row in plan.get("objects", [])} != trainval_ids
            or plan.get("dataset_revision") != revision):
        raise ValueError("Train/validation plan and split manifest disagree")
    development_ids = DEV_RANGE | previous_test_ids
    excluded_ids = development_ids | trainval_ids

    excluded_features = {"name_key": set(), "source_key": set(), "geometry_sha256": set()}
    for object_id in sorted(excluded_ids):
        features = _metadata(root / "data" / object_id)
        for key in excluded_features:
            excluded_features[key].add(features[key])

    candidates, rejected = [], {}
    for folder in sorted((root / "data").glob("UPB_*")):
        if not folder.is_dir() or not OBJECT_RE.fullmatch(folder.name) or folder.name in excluded_ids:
            continue
        ok, category, reason = _eligible(folder)
        if not ok:
            rejected[reason or "ineligible"] = rejected.get(reason or "ineligible", 0) + 1
            continue
        try:
            features = _metadata(folder)
        except (OSError, KeyError, TypeError, ValueError):
            rejected["invalid_near_duplicate_metadata"] = (
                rejected.get("invalid_near_duplicate_metadata", 0) + 1)
            continue
        if features["category"] != category:
            rejected["category_mismatch"] = rejected.get("category_mismatch", 0) + 1
            continue
        candidates.append({"id": folder.name, **features})

    candidates.sort(key=lambda row: hashlib.sha256(
        f"{seed}:{revision}:{row['id']}".encode()).hexdigest())
    chosen, chosen_features = [], {key: set() for key in excluded_features}
    checked = 0
    for row in candidates:
        checked += 1
        duplicate_key = next((key for key in excluded_features
                              if row[key] in excluded_features[key]
                              or row[key] in chosen_features[key]), None)
        if duplicate_key:
            reason = f"near_duplicate_{duplicate_key}"
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        ok, reason = _convertible(root / "data" / row["id"], row["category"])
        if not ok:
            rejected[reason or "conversion_unknown"] = rejected.get(
                reason or "conversion_unknown", 0) + 1
            continue
        chosen.append({"id": row["id"], "category": row["category"], "split": "test",
                       "source_identity_sha256": hashlib.sha256(row["source_key"].encode()).hexdigest(),
                       "geometry_sha256": row["geometry_sha256"]})
        for key in chosen_features:
            chosen_features[key].add(row[key])
        if len(chosen) == objects:
            break
    if len(chosen) != objects:
        raise ValueError(f"Need {objects} unseen, non-duplicate test objects, found {len(chosen)}")

    record = {
        "version": 1,
        "role": "formal_test",
        "dataset": DATASET,
        "dataset_revision": revision,
        "license": LICENSE,
        "seed": seed,
        "expected_object_counts": {"test": objects},
        "previous_test_splits_sha256": _sha256(previous_test_splits),
        "trainval_splits_sha256": _sha256(trainval_splits),
        "trainval_plan_sha256": _sha256(trainval_plan),
        "excluded_development_object_ids": sorted(development_ids),
        "excluded_train_validation_count": len(trainval_ids),
        "near_duplicate_policy": ["source_identity", "full_model_sha256", "normalized_object_name"],
        "candidates_checked": checked,
        "rejected": rejected,
        "objects": chosen,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/assets/uniphys"))
    parser.add_argument("--previous-test-splits", type=Path,
                        default=Path("data/assets/uniphys-dev-catalog/test-splits.json"))
    parser.add_argument("--trainval-splits", type=Path,
                        default=Path("data/assets/uniphys-trainval-catalog-v1/trainval-splits.json"))
    parser.add_argument("--trainval-plan", type=Path, default=Path("configs/uniphys-trainval-v1.json"))
    parser.add_argument("--out", type=Path, default=Path("configs/uniphys-test.json"))
    parser.add_argument("--objects", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    result = select(args.root.resolve(), args.previous_test_splits.resolve(),
                    args.trainval_splits.resolve(), args.trainval_plan.resolve(),
                    args.out.resolve(), objects=args.objects, seed=args.seed)
    print(json.dumps({"status": "complete", "objects": len(result["objects"]),
                      "output": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
