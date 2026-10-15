#!/usr/bin/env python3
"""Aggregate v5 stopping runs and paired exact comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mcnemar(left: list[dict], right: list[dict]) -> dict:
    left_by_id = {row["id"]: row for row in left}
    right_by_id = {row["id"]: row for row in right}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("Paired comparison requires identical case IDs")
    left_only = right_only = 0
    for uid in left_by_id:
        a, b = left_by_id[uid], right_by_id[uid]
        a_ok = a["predicted_decision"] == a["expected_decision"]
        b_ok = b["predicted_decision"] == b["expected_decision"]
        left_only += a_ok and not b_ok
        right_only += b_ok and not a_ok
    discordant = left_only + right_only
    tail = min(left_only, right_only)
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)) \
        if discordant else 1.0
    return {
        "left_correct_right_wrong": left_only,
        "right_correct_left_wrong": right_only,
        "discordant": discordant,
        "two_sided_exact_p": p,
        "test": "exact McNemar/binomial on per-case decision correctness",
    }


def _compact(metrics: dict) -> dict:
    overall, pairs = metrics["overall"], metrics["strict_pairs"]
    return {
        "n": overall["n"], "accuracy": overall["accuracy"],
        "stop_precision": overall["stop_precision"], "stop_recall": overall["stop_recall"],
        "premature_stop": overall["premature_stop"], "missed_stop": overall["missed_stop"],
        "active_stop_recall": overall["active_stop"]["decision_recall"],
        "forced_stop_recall": overall["forced_budget_stop"]["decision_recall"],
        "strict_pair_correct": pairs["correctly_distinguished"],
        "strict_pair_n": pairs["n"],
        "strict_pair_rate": pairs["correctly_distinguished_rate"],
        "strict_pair_nonvisual_equal": pairs["all_model_visible_nonimage_input_equal"],
    }


def compare(inputs: dict[str, Path], output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    metrics = {name: json.loads((path / "metrics.json").read_text()) for name, path in inputs.items()}
    predictions = {name: _read(path / "predictions.jsonl") for name, path in inputs.items()}
    v5 = predictions["v5"]
    report = {
        "version": 1, "status": "complete", "scope": "v5_visual_stopping_comparison",
        "table": {name: _compact(value) for name, value in metrics.items()},
        "paired_decision_tests": {
            f"v5_vs_{name}": _mcnemar(v5, rows)
            for name, rows in predictions.items() if name != "v5"
        },
        "inputs": {
            name: {"directory": str(path),
                   "metrics_sha256": _sha256(path / "metrics.json"),
                   "predictions_sha256": _sha256(path / "predictions.jsonl")}
            for name, path in inputs.items()
        },
        "statistical_scope": (
            "Exact tests pair the 389 held-out rows. This is one training seed and seven validation "
            "objects; row-level p-values do not establish across-seed or across-dataset significance."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--v4", type=Path, required=True)
    parser.add_argument("--v5", type=Path, required=True)
    parser.add_argument("--masked", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = {"base": args.base.resolve(), "v4_evidence_round2": args.v4.resolve(),
              "v5": args.v5.resolve(), "v5_masked": args.masked.resolve(),
              "no_image_budget": args.budget.resolve()}
    print(json.dumps(compare(inputs, args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
