#!/usr/bin/env python3
"""Audit fixed-evidence replay identity and compare it with a baseline run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUNS = ("base-train", "base-heldout", "sft-train", "sft-heldout", "pixel-baseline")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prediction_identity(row: dict) -> dict:
    """Exclude runtime-only timing; retain every model-visible/result field."""
    return {key: value for key, value in row.items() if key != "seconds"}


def main(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError("Output must be fresh")
    replay = json.loads((args.replay / "data" / "replay_manifest.json").read_text())
    source_inputs = args.source / "inputs"
    failures = []
    for case in replay["cases"]:
        case_id = case["case_id"]
        actual = args.replay / "data" / case["image"]
        source = source_inputs / f"{case_id}.png"
        if sha256(actual) != sha256(source):
            failures.append(f"image_sha_mismatch:{case_id}")
        if case["sheet_size"] != [542, 542] or case["position_order"] != ["ref-rest", "ref-limit", "candidate-rest", "candidate-limit"]:
            failures.append(f"layout_mismatch:{case_id}")
        if any(not tile["size"] == [256, 256] for tile in case["tile_info"]):
            failures.append(f"tile_size_mismatch:{case_id}")
        if not case["original_comparison"]["byte_equal"]:
            failures.append(f"pixel_mismatch:{case_id}")
    predictions = {}
    for name in RUNS:
        new = rows(args.replay / name / "predictions.jsonl")
        old = rows(args.baseline / name / "predictions.jsonl")
        if [prediction_identity(row) for row in new] != [prediction_identity(row) for row in old]:
            failures.append(f"prediction_mismatch:{name}")
        predictions[name] = {"rows": len(new), "semantic_output_identical_to_baseline":
                             [prediction_identity(row) for row in new] == [prediction_identity(row) for row in old]}
    if sha256(args.replay / "data" / "split.json") != sha256(args.baseline / "data" / "split.json"):
        failures.append("split_mismatch")
    report = {"status": "passed" if not failures else "failed", "cases": len(replay["cases"]),
              "all_replayed_inputs_byte_identical": not any(x.startswith(("image_", "pixel_", "layout_", "tile_")) for x in failures),
              "predictions": predictions, "failures": failures}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
