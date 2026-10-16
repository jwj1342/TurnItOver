"""Deterministic leakage, split, balance, image, and evidence audit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from turnitover.oracle.motion_diagnosis import validate_target


def main(root: Path, out: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    rows = {split: [json.loads(x) for x in (root / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "validation")}
    errors = []
    assets = {split: {row["asset_id"] for row in values} for split, values in rows.items()}
    if assets["train"] & assets["validation"]:
        errors.append("train_validation_asset_overlap")
    for split, values in rows.items():
        groups = {}
        for row in values:
            try:
                validate_target(row["ordinary_target"]); validate_target(row["evidence_target"])
            except ValueError as exc:
                errors.append(f"{row['id']}:{exc}")
            if any(key in row["messages"][0]["content"] for key in ("candidate_sha", "latent_class", "gold")):
                errors.append(f"{row['id']}:private_metadata_in_prompt")
            for relative, expected in zip(row["images"], row["image_sha256"], strict=True):
                path = root / relative
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    errors.append(f"{row['id']}:image_sha")
            groups.setdefault((row["asset_id"], row["condition"]), []).append(row)
        for asset in assets[split]:
            full = groups.get((asset, "full_motion"), [])
            rest = groups.get((asset, "rest_only"), [])
            if Counter(r["gold"]["class"] for r in full) != Counter(axis=1, offset=1, clean=1) or len(rest) != 1:
                errors.append(f"{asset}:unbalanced_triplet_or_uncertain")
            if full and len({r["messages"][0]["content"] for r in full}) != 1:
                errors.append(f"{asset}:class_dependent_full_prompt")
        for row in values:
            if row["condition"] == "rest_only" and row["evidence_target"]["defect_type"] != "uncertain":
                errors.append(f"{row['id']}:insufficient_not_uncertain")
            if row["condition"] == "full_motion" and row["evidence_target"]["defect_type"] in ("axis", "offset"):
                steps = row["evidence_target"]["evidence_steps"]
                if not steps or not set(steps) <= set(row["gold"]["perceptible_steps"]):
                    errors.append(f"{row['id']}:unsupported_evidence_steps")
    report = {"status": "passed" if not errors else "failed", "errors": errors,
              "train_objects": len(assets["train"]), "validation_objects": len(assets["validation"]),
              "train_rows": len(rows["train"]), "validation_rows": len(rows["validation"]),
              "same_scale": len(rows["train"]), "frozen_test_used": manifest.get("frozen_test_used")}
    out.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); main(args.root.resolve(), args.out.resolve())
