#!/usr/bin/env python3
"""Freeze disjoint UniPhys object IDs for train and validation case generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.core.jsonio import write_json_atomic
from turnitover.assets.urdf import UrdfModel


DATASET = "breezexian/UniPhys-Bench-Part2"
LICENSE = "CC-BY-NC-4.0"
OBJECT_RE = re.compile(r"UPB_\d{8}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_ids(path: Path) -> set[str]:
    record = json.loads(path.read_text(encoding="utf-8"))
    groups = record.get("groups")
    if record.get("version") != 1 or not isinstance(groups, dict) or not groups:
        raise ValueError("Expected a nonempty version-1 test split manifest")
    result = set()
    for group, split in groups.items():
        object_id = group.rsplit(":", 1)[-1] if isinstance(group, str) else ""
        if split != "test" or not OBJECT_RE.fullmatch(object_id):
            raise ValueError("Test split manifest contains an invalid group")
        result.add(object_id)
    return result


def _eligible(folder: Path) -> tuple[bool, str | None, str | None]:
    """Cheap structural preflight before the more expensive mesh conversion."""
    try:
        root = ET.parse(folder / "model.urdf").getroot()
        metadata = json.loads((folder / "meta_data.json").read_text(encoding="utf-8"))
        category = metadata["object"]["category"]
        if root.tag != "robot" or not isinstance(category, str) or not category:
            return False, None, "invalid_metadata_or_urdf"
        movable = 0
        for joint in root.findall("joint"):
            joint_type = joint.get("type")
            if joint.find("mimic") is not None or joint_type not in {
                    "fixed", "continuous", "revolute", "prismatic"}:
                return False, category, "unsupported_joint"
            if joint_type not in {"revolute", "prismatic"}:
                continue
            limit = joint.find("limit")
            if limit is None:
                return False, category, "missing_joint_limit"
            lower, upper = float(limit.get("lower", "nan")), float(limit.get("upper", "nan"))
            if not math.isfinite(lower) or not math.isfinite(upper) or not lower <= 0 <= upper or lower == upper:
                return False, category, "invalid_joint_limit"
            movable += 1
        return (True, category, None) if movable else (False, category, "no_finite_joint")
    except (OSError, KeyError, TypeError, ValueError, ET.ParseError):
        return False, None, "invalid_asset"


def _convertible(folder: Path, category: str) -> tuple[bool, str | None]:
    try:
        spec = UrdfModel(folder / "model.urdf", continuous_joint_policy="fixed").spec(
            folder.name, category)
        return (True, None) if spec.joints else (False, "no_finite_joint_after_conversion")
    except Exception as exc:
        return False, f"conversion_{type(exc).__name__}"


def select(root: Path, test_splits: Path, output: Path, *, train_objects: int = 100,
           validation_objects: int = 20, seed: int = 20261011) -> dict:
    if output.exists():
        raise ValueError(f"Selection output already exists: {output}")
    if train_objects < 1 or validation_objects < 1:
        raise ValueError("Train and validation object counts must be positive")
    acquisition = json.loads((root / "download-manifest.json").read_text(encoding="utf-8"))
    revision = acquisition.get("revision")
    if (acquisition.get("dataset") != DATASET or acquisition.get("license") != LICENSE
            or not isinstance(revision, str) or len(revision) != 40):
        raise ValueError("Expected a pinned UniPhys-Bench Part 2 acquisition")

    held_out = _test_ids(test_splits)
    candidates, rejected = [], {}
    for folder in sorted((root / "data").glob("UPB_*")):
        if not folder.is_dir() or not OBJECT_RE.fullmatch(folder.name) or folder.name in held_out:
            continue
        ok, category, reason = _eligible(folder)
        if ok:
            candidates.append({"id": folder.name, "category": category})
        else:
            rejected[reason or "unknown"] = rejected.get(reason or "unknown", 0) + 1

    needed = train_objects + validation_objects
    if len(candidates) < needed:
        raise ValueError(f"Need {needed} eligible non-test objects, found {len(candidates)}")
    candidates.sort(key=lambda row: hashlib.sha256(
        f"{seed}:{revision}:{row['id']}".encode()).hexdigest())
    chosen = []
    checked = 0
    for row in candidates:
        checked += 1
        ok, reason = _convertible(root / "data" / row["id"], row["category"])
        if ok:
            chosen.append(row)
            if len(chosen) == needed:
                break
        else:
            rejected[reason or "conversion_unknown"] = rejected.get(
                reason or "conversion_unknown", 0) + 1
    if len(chosen) < needed:
        raise ValueError(f"Need {needed} fully convertible objects, found {len(chosen)}")
    objects = [dict(row, split="validation" if index < validation_objects else "train")
               for index, row in enumerate(chosen)]
    record = {
        "version": 1,
        "dataset": DATASET,
        "dataset_revision": revision,
        "license": LICENSE,
        "seed": seed,
        "test_splits_sha256": _sha256(test_splits),
        "test_object_ids": sorted(held_out),
        "expected_object_counts": {"train": train_objects, "validation": validation_objects},
        "eligible_non_test_objects": len(candidates),
        "conversion_candidates_checked": checked,
        "rejected": rejected,
        "objects": objects,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/assets/uniphys"))
    parser.add_argument("--test-splits", type=Path,
                        default=Path("data/assets/uniphys-dev-catalog/test-splits.json"))
    parser.add_argument("--out", type=Path, default=Path("configs/uniphys-trainval-v1.json"))
    parser.add_argument("--train-objects", type=int, default=100)
    parser.add_argument("--validation-objects", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    result = select(args.root.resolve(), args.test_splits.resolve(), args.out.resolve(),
                    train_objects=args.train_objects, validation_objects=args.validation_objects,
                    seed=args.seed)
    print(json.dumps({"status": "complete", "eligible": result["eligible_non_test_objects"],
                      "counts": result["expected_object_counts"], "output": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
