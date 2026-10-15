#!/usr/bin/env python3
"""Aggregate comparable Episode metrics for three modular-verifier systems."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prf(expected: list[set], predicted: list[set]) -> dict:
    tp = sum(len(left & right) for left, right in zip(expected, predicted, strict=True))
    fp = sum(len(right - left) for left, right in zip(expected, predicted, strict=True))
    fn = sum(len(left - right) for left, right in zip(expected, predicted, strict=True))
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def _signatures(row: dict) -> tuple[set, set]:
    expected = {(item[0], item[1], tuple(item[2])) for item in row["expected_visible_findings"]}
    predicted = {(item["defect_id"], item["joint_id"], tuple(sorted(item["parts"])))
                 for item in row["accepted_findings"]}
    return expected, predicted


def _aggregate(rows: list[dict]) -> dict:
    pairs = [_signatures(row) for row in rows]
    expected = [pair[0] for pair in pairs]
    predicted = [pair[1] for pair in pairs]
    expected_defects = [{item[0] for item in values} for values in expected]
    predicted_defects = [{item[0] for item in values} for values in predicted]
    expected_parts = [{(item[0], item[2]) for item in values} for values in expected]
    predicted_parts = [{(item[0], item[2]) for item in values} for values in predicted]
    fail_rows = [row for row in rows if row["final_status"] == "fail"]
    timing = {}
    for row in rows:
        timing[row["stop_timing"]] = timing.get(row["stop_timing"], 0) + 1
    return {
        "episodes": len(rows), "assets": len({row["asset_id"] for row in rows}),
        "status_accuracy": sum(
            row["final_status"] == ("fail" if expected[index] else "uncertain")
            for index, row in enumerate(rows)) / len(rows),
        "finding_micro": _prf(expected, predicted),
        "defect_type_micro": _prf(expected_defects, predicted_defects),
        "part_localization_micro": _prf(expected_parts, predicted_parts),
        "unsupported_findings": sum(row["unsupported_findings"] for row in rows),
        "coordinator_rejected_findings": sum(
            len(call["rejected_findings"]) for row in rows for call in row["calls"]),
        "premature_stop": timing.get("early", 0),
        "missed_stop": timing.get("late", 0) + timing.get("missed", 0),
        "stop_timing": timing,
        "evidence_sufficient_fail_stops": sum(
            row["evidence_sufficient_at_fail"] is True for row in fail_rows),
        "fail_stops": len(fail_rows),
        "evidence_sufficient_stop_rate": (
            sum(row["evidence_sufficient_at_fail"] is True for row in fail_rows) / len(fail_rows)
            if fail_rows else None),
        "uncertain": sum(row["final_status"] == "uncertain" for row in rows),
        "average_observations": sum(row["observation_count"] for row in rows) / len(rows),
        "clean_false_fail": sum(row["gold_stratum"] == "clean" and row["final_status"] == "fail"
                                for row in rows),
        "v4_unmapped_observations": sum(row["v4_unmapped_observations"] for row in rows),
    }


def _key(row: dict) -> tuple[str, str]:
    return row["source_sample_id"], row["policy"]


def audit(args) -> dict:
    paths = {"v4_only": args.v4_only, "modular_v4_v5": args.modular,
             "failed_multitask_lora": args.multitask}
    rows = {name: _read_jsonl(path / "episodes.jsonl") for name, path in paths.items()}
    key_sets = [{_key(row) for row in values} for values in rows.values()]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Systems do not cover identical Episodes")
    for key in key_sets[0]:
        traces = [next(row for row in values if _key(row) == key) for values in rows.values()]
        if len({(row["episode_id"], row["expected_first_stop_step"]) for row in traces}) != 1:
            raise ValueError(f"Systems differ in policy or scoring target: {key}")
    table = {name: _aggregate(values) for name, values in rows.items()}
    run_metrics = {name: json.loads((path / "metrics.json").read_text())
                   for name, path in paths.items()}
    modular, v4, multi = table["modular_v4_v5"], table["v4_only"], table["failed_multitask_lora"]
    failures = []
    for row in rows["modular_v4_v5"]:
        if row["unsupported_findings"] or row["stop_timing"] in ("early", "late", "missed"):
            failures.append({
                "episode_id": row["episode_id"], "policy": row["policy"],
                "asset_id": row["asset_id"], "stratum": row["gold_stratum"],
                "observations": row["observation_count"], "timing": row["stop_timing"],
                "status": row["final_status"], "unsupported": row["unsupported_findings"],
                "expected": row["expected_visible_findings"],
                "predicted": row["accepted_findings"],
            })
    report = {
        "version": 1, "status": "complete", "scope": "modular_verifier_comparison",
        "raw_table": table,
        "per_policy": {name: {policy: _aggregate([row for row in values if row["policy"] == policy])
                              for policy in ("fixed", "random")}
                       for name, values in rows.items()},
        "per_asset": {name: {asset: _aggregate([row for row in values if row["asset_id"] == asset])
                             for asset in sorted({row["asset_id"] for row in values})}
                      for name, values in rows.items()},
        "deltas": {
            "modular_minus_v4_only": {
                "status_accuracy": modular["status_accuracy"] - v4["status_accuracy"],
                "finding_f1": modular["finding_micro"]["f1"] - v4["finding_micro"]["f1"],
                "defect_type_f1": modular["defect_type_micro"]["f1"] - v4["defect_type_micro"]["f1"],
                "part_localization_f1": (modular["part_localization_micro"]["f1"]
                                         - v4["part_localization_micro"]["f1"]),
                "unsupported_findings": modular["unsupported_findings"] - v4["unsupported_findings"],
                "premature_stop": modular["premature_stop"] - v4["premature_stop"],
                "missed_stop": modular["missed_stop"] - v4["missed_stop"],
                "average_observations": modular["average_observations"] - v4["average_observations"],
            },
            "modular_minus_failed_multitask_lora": {
                "status_accuracy": modular["status_accuracy"] - multi["status_accuracy"],
                "finding_f1": modular["finding_micro"]["f1"] - multi["finding_micro"]["f1"],
                "unsupported_findings": modular["unsupported_findings"] - multi["unsupported_findings"],
                "premature_stop": modular["premature_stop"] - multi["premature_stop"],
                "missed_stop": modular["missed_stop"] - multi["missed_stop"],
                "average_observations": modular["average_observations"] - multi["average_observations"],
            },
        },
        "modular_failure_cases": failures,
        "compatibility": {
            "v4_unmapped_observations": modular["v4_unmapped_observations"],
            "cause": "v5 left/limit aligned states have no exact entry in the frozen v4 exposure grid",
            "handling": "kept visible to models but excluded from defect-qualified post-hoc evidence; no label projection",
        },
        "inputs": {name: {"directory": str(path),
                           "metrics_sha256": _sha(path / "metrics.json"),
                           "episodes_sha256": _sha(path / "episodes.jsonl"),
                           "stop_checkpoint_sha256": _sha(
                               Path(run_metrics[name]["stop_adapter"]) / "adapter_model.safetensors"),
                           "diagnosis_checkpoint_sha256": _sha(
                               Path(run_metrics[name]["diagnosis_adapter"]) / "adapter_model.safetensors")}
                   for name, path in paths.items()},
        "code_sha256": {
            "evaluator": _sha(REPO_ROOT / "scripts/evaluate_modular_verifier.py"),
            "coordinator": _sha(REPO_ROOT / "turnitover/verifier/modular.py"),
        },
        "artifact_integrity": {
            name: {key: run_metrics[name][key]
                   for key in ("all_reference_sha_match", "all_v4_single_reference_sha_match",
                               "all_observation_sha_match", "frozen_test_used")}
            for name, path in paths.items()
        },
        "reproduction": {
            "seed": 20261017, "episodes_per_system": 14,
            "systems": {
                "v4_only": "v4 stop + v4 diagnosis; one-reference protocols",
                "modular_v4_v5": "v5 eight-reference stopping + v4 one-reference diagnosis",
                "failed_multitask_lora": "failed multitask adapter used for both separated protocols",
            },
        },
        "decision": {
            "roadmap_system_baseline": False,
            "reason": "Stopping safety improves, but diagnosis/localization and protocol compatibility remain inadequate.",
            "next_priority": "dynamic evidence-to-diagnosis grounding before learned action selection or repair",
        },
        "limitations": [
            "Only seven held-out objects and fourteen Episodes per system are evaluated.",
            "Fixed/random schedules are predetermined and are not active policies.",
            "Selection uses private labels only for predeclared coverage; all online decisions are label-free.",
            "No pass-qualified Episode or natural error with independent human Gold is available.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-only", type=Path, required=True)
    parser.add_argument("--modular", type=Path, required=True)
    parser.add_argument("--multitask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
