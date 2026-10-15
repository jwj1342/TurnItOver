#!/usr/bin/env python3
"""Audit whether one adapter retains v4 diagnosis and v5 visual stopping."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_diagnosis(row: dict) -> bool:
    return (row["predicted_status"] == row["expected_status"]
            and sorted(row["predicted_findings"]) == sorted(row["expected_findings"]))


def _mcnemar(left: dict[str, bool], right: dict[str, bool]) -> dict:
    if left.keys() != right.keys():
        raise ValueError("Paired tests require identical case IDs")
    left_only = sum(left[key] and not right[key] for key in left)
    right_only = sum(right[key] and not left[key] for key in left)
    n = left_only + right_only
    tail = min(left_only, right_only)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(tail + 1)) / 2 ** n) if n else 1.0
    return {"left_correct_right_wrong": left_only,
            "right_correct_left_wrong": right_only,
            "discordant": n, "two_sided_exact_p": p}


def _diagnosis_table(metrics: dict) -> dict:
    return {
        "n": metrics["n"], "valid_output_rate": metrics["valid_output_rate"],
        "status_accuracy": metrics["status_accuracy"],
        "exact_verdict_accuracy": metrics["exact_verdict_accuracy"],
        "finding_precision": metrics["finding_micro"]["precision"],
        "finding_recall": metrics["finding_micro"]["recall"],
        "finding_f1": metrics["finding_micro"]["f1"],
        "defect_type_f1": metrics["defect_type_micro"]["f1"],
        "part_localization_f1": metrics["part_localization_micro"]["f1"],
        "unsupported_findings": metrics["evidence_consistency"]["unsupported_findings"],
        "predicted_findings": metrics["evidence_consistency"]["predicted_findings"],
        "per_asset": metrics["per_asset"],
    }


def _stopping_table(metrics: dict) -> dict:
    overall, pairs = metrics["overall"], metrics["strict_pairs"]
    return {
        "n": overall["n"], "accuracy": overall["accuracy"],
        "stop_precision": overall["stop_precision"], "stop_recall": overall["stop_recall"],
        "premature_stop": overall["premature_stop"], "missed_stop": overall["missed_stop"],
        "active_stop_recall": overall["active_stop"]["decision_recall"],
        "forced_stop_recall": overall["forced_budget_stop"]["decision_recall"],
        "strict_pair_correct": pairs["correctly_distinguished"],
        "strict_pair_n": pairs["n"], "strict_pair_rate": pairs["correctly_distinguished_rate"],
        "per_step": metrics["per_step"],
    }


def _stopping_per_asset(rows: list[dict]) -> dict:
    buckets = defaultdict(lambda: {"n": 0, "correct": 0, "premature_stop": 0,
                                   "missed_stop": 0, "active_stop_n": 0,
                                   "active_stop_correct": 0, "forced_stop_n": 0,
                                   "forced_stop_correct": 0})
    for row in rows:
        bucket = buckets[row["asset_id"]]
        expected, predicted = row["expected_decision"], row["predicted_decision"]
        bucket["n"] += 1
        bucket["correct"] += predicted == expected
        bucket["premature_stop"] += expected == "continue" and predicted == "stop"
        bucket["missed_stop"] += expected == "stop" and predicted != "stop"
        if expected == "stop":
            prefix = "forced_stop" if row["termination_kind"] == "forced" else "active_stop"
            bucket[f"{prefix}_n"] += 1
            bucket[f"{prefix}_correct"] += predicted == "stop"
    return {asset: dict(values, accuracy=values["correct"] / values["n"])
            for asset, values in sorted(buckets.items())}


def audit(args) -> dict:
    diagnosis_paths = {"v4_evidence_round2": args.v4_diagnosis,
                        "v5_stopping_only": args.v5_diagnosis,
                        "multitask": args.multitask_diagnosis}
    stopping_paths = {"v4_evidence_round2": args.v4_stopping,
                       "v5_stopping_only": args.v5_stopping,
                       "multitask": args.multitask_stopping}
    diagnosis_metrics = {key: json.loads((path / "metrics.json").read_text())
                         for key, path in diagnosis_paths.items()}
    stopping_metrics = {key: json.loads((path / "metrics.json").read_text())
                        for key, path in stopping_paths.items()}
    diagnosis_rows = {key: _read(path / "predictions.jsonl") for key, path in diagnosis_paths.items()}
    stopping_rows = {key: _read(path / "predictions.jsonl") for key, path in stopping_paths.items()}
    v4_diag = {row["id"]: _exact_diagnosis(row) for row in diagnosis_rows["v4_evidence_round2"]}
    multi_diag = {row["id"]: _exact_diagnosis(row) for row in diagnosis_rows["multitask"]}
    v5_stop = {row["id"]: row["predicted_decision"] == row["expected_decision"]
               for row in stopping_rows["v5_stopping_only"]}
    multi_stop = {row["id"]: row["predicted_decision"] == row["expected_decision"]
                  for row in stopping_rows["multitask"]}
    train_run = json.loads(args.train_run.read_text())
    closed_loop = json.loads((args.closed_loop / "metrics.json").read_text())
    diag_v4 = _diagnosis_table(diagnosis_metrics["v4_evidence_round2"])
    diag_multi = _diagnosis_table(diagnosis_metrics["multitask"])
    stop_v5 = _stopping_table(stopping_metrics["v5_stopping_only"])
    stop_multi = _stopping_table(stopping_metrics["multitask"])
    report = {
        "version": 1, "status": "complete", "scope": "v4_diagnosis_v5_stopping_integration",
        "diagnosis_table": {key: _diagnosis_table(value) for key, value in diagnosis_metrics.items()},
        "stopping_table": {
            key: _stopping_table(value) | {"per_asset": _stopping_per_asset(stopping_rows[key])}
            for key, value in stopping_metrics.items()
        },
        "deltas": {
            "multitask_minus_v4_diagnosis": {
                key: diag_multi[key] - diag_v4[key] for key in (
                    "status_accuracy", "exact_verdict_accuracy", "finding_f1",
                    "defect_type_f1", "part_localization_f1")
            } | {"unsupported_findings": diag_multi["unsupported_findings"] - diag_v4["unsupported_findings"]},
            "multitask_minus_v5_stopping": {
                key: stop_multi[key] - stop_v5[key] for key in (
                    "accuracy", "stop_precision", "stop_recall", "active_stop_recall",
                    "forced_stop_recall", "strict_pair_rate")
            } | {"premature_stop": stop_multi["premature_stop"] - stop_v5["premature_stop"],
                 "missed_stop": stop_multi["missed_stop"] - stop_v5["missed_stop"]},
        },
        "paired_tests": {
            "multitask_vs_v4_exact_diagnosis": _mcnemar(multi_diag, v4_diag),
            "multitask_vs_v5_stop_decision": _mcnemar(multi_stop, v5_stop),
        },
        "closed_loop": closed_loop,
        "retention_gate": {
            "diagnosis_f1_drop_at_most_0.05": diag_multi["finding_f1"] >= diag_v4["finding_f1"] - .05,
            "unsupported_findings_not_increased": diag_multi["unsupported_findings"] <= diag_v4["unsupported_findings"],
            "stopping_accuracy_drop_at_most_0.05": stop_multi["accuracy"] >= stop_v5["accuracy"] - .05,
            "strict_pair_drop_at_most_0.05": stop_multi["strict_pair_rate"] >= stop_v5["strict_pair_rate"] - .05,
            "all_active_browser_stops_supported": closed_loop["active_stop_support_rate"] == 1.0,
        },
        "train": {"path": str(args.train_run), "sha256": _sha(args.train_run),
                  "status": train_run["status"], "mixing": train_run["mixing"]},
        "inputs": {
            "diagnosis": {key: {"directory": str(path),
                                  "metrics_sha256": _sha(path / "metrics.json"),
                                  "predictions_sha256": _sha(path / "predictions.jsonl")}
                           for key, path in diagnosis_paths.items()},
            "stopping": {key: {"directory": str(path),
                                 "metrics_sha256": _sha(path / "metrics.json"),
                                 "predictions_sha256": _sha(path / "predictions.jsonl")}
                          for key, path in stopping_paths.items()},
            "closed_loop": {"directory": str(args.closed_loop),
                              "metrics_sha256": _sha(args.closed_loop / "metrics.json")},
        },
        "comparison_limits": [
            "v4 diagnosis and v5 stopping use different reference-image protocols and are separate tasks.",
            "The browser smoke uses fixed actions; no learned action selection is tested.",
            "One seed and seven held-out objects do not establish cross-seed generalization.",
        ],
        "frozen_test_used": False,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v4_diagnosis", "v5_diagnosis", "multitask_diagnosis",
                 "v4_stopping", "v5_stopping", "multitask_stopping", "closed_loop"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
