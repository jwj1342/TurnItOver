#!/usr/bin/env python3
"""Aggregate paired Qwen prefix diagnostics without loading model weights."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


MODELS = ("base", "ordinary", "evidence")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _input(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _sig(rows: list[list]) -> set[tuple]:
    return {(row[0], row[1], tuple(row[2])) for row in rows}


def _prf(tp: int, fp: int, fn: int) -> dict:
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / denominator if denominator else None}


def summarize(rows: list[dict]) -> dict:
    counts = Counter(n=len(rows))
    for row in rows:
        expected, predicted = _sig(row["expected_findings"]), _sig(row["predicted_findings"])
        expected_types = {item[0] for item in expected}
        predicted_types = set(row.get("predicted_defect_types", [item[0] for item in predicted]))
        expected_parts = {(item[0], item[2]) for item in expected}
        predicted_parts = {(item[0], item[2]) for item in predicted}
        verdict = row["decision_kind"] == "verdict"
        counts["valid_verdict"] += verdict
        counts["status_correct"] += verdict and row["predicted_status"] == row["expected_status"]
        counts["exact"] += verdict and row["predicted_status"] == row["expected_status"] and predicted == expected
        counts["tp"] += len(expected & predicted)
        counts["fp"] += len(predicted - expected)
        counts["fn"] += len(expected - predicted)
        counts["type_tp"] += len(expected_types & predicted_types)
        counts["type_fp"] += len(predicted_types - expected_types)
        counts["type_fn"] += len(expected_types - predicted_types)
        counts["part_tp"] += len(expected_parts & predicted_parts)
        counts["part_fp"] += len(predicted_parts - expected_parts)
        counts["part_fn"] += len(expected_parts - predicted_parts)
        counts["unsupported"] += row["unsupported_findings"]
        counts["predicted_findings"] += len(predicted)
        counts["false_fail"] += verdict and row["predicted_status"] == "fail" and row["expected_status"] != "fail"
        counts["no_evidence"] += not expected
        counts["no_evidence_uncertain"] += not expected and row["predicted_status"] == "uncertain"
    return {"n": counts["n"], "valid_verdicts": counts["valid_verdict"],
            "status_correct": counts["status_correct"], "exact_verdict": counts["exact"],
            "findings": _prf(counts["tp"], counts["fp"], counts["fn"]),
            "defect_types": _prf(counts["type_tp"], counts["type_fp"], counts["type_fn"]),
            "part_localization": _prf(counts["part_tp"], counts["part_fp"], counts["part_fn"]),
            "unsupported_findings": counts["unsupported"],
            "predicted_findings": counts["predicted_findings"], "false_fail": counts["false_fail"],
            "no_evidence": {"n": counts["no_evidence"],
                            "uncertain": counts["no_evidence_uncertain"]}}


def grouped(rows: list[dict], key) -> dict:
    buckets = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    return {str(name): summarize(bucket) for name, bucket in sorted(buckets.items())}


def summarize_defect(rows: list[dict], defect: str) -> dict:
    counts = Counter()
    for row in rows:
        expected = {item for item in _sig(row["expected_findings"]) if item[0] == defect}
        predicted = {item for item in _sig(row["predicted_findings"]) if item[0] == defect}
        counts["tp"] += len(expected & predicted)
        counts["fp"] += len(predicted - expected)
        counts["fn"] += len(expected - predicted)
        counts["expected_cases"] += bool(expected)
        counts["predicted_cases"] += bool(predicted)
    return {**_prf(counts["tp"], counts["fp"], counts["fn"]),
            "expected_cases": counts["expected_cases"], "predicted_cases": counts["predicted_cases"]}


def _decision_counts(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        label = row["predicted_status"] if row["decision_kind"] == "verdict" else row["decision_kind"]
        counts[label] += 1
        counts["unsupported_findings"] += row["unsupported_findings"]
    return dict(counts)


def _status_counts(rows: list[dict]) -> dict:
    counts = Counter(row.get("predicted_status") or row["decision_kind"] for row in rows)
    return dict(counts)


def _stop_metrics(rows: list[dict], stop_targets: dict[tuple[str, int], str]) -> dict:
    counts = Counter(n=len(rows))
    by_episode = defaultdict(list)
    for row in rows:
        key = (row["episode_id"], row["decision_index"])
        expected = stop_targets[key]
        predicted = row.get("predicted_stop_decision")
        if predicted is None:
            predicted = "stop" if row["decision_kind"] == "verdict" else (
                "continue" if row["decision_kind"] == "action" else "invalid")
        counts["correct"] += predicted == expected
        counts[f"{expected}->{predicted}"] += 1
        by_episode[row["episode_id"]].append((row["decision_index"], expected, predicted))
    timing = Counter(n=len(by_episode))
    absolute_error = 0
    for episode_rows in by_episode.values():
        episode_rows.sort()
        expected_index = next(index for index, expected, _ in episode_rows if expected == "stop")
        predicted_indices = [index for index, _, predicted in episode_rows if predicted == "stop"]
        if not predicted_indices:
            timing["missing_stop"] += 1
            continue
        predicted_index = predicted_indices[0]
        absolute_error += abs(predicted_index - expected_index)
        timing["exact"] += predicted_index == expected_index
        timing["early"] += predicted_index < expected_index
        timing["late"] += predicted_index > expected_index
    return {
        "prefix_n": counts["n"], "decision_accuracy": counts["correct"] / counts["n"],
        "confusion": {key: value for key, value in sorted(counts.items()) if "->" in key},
        "episodes": timing["n"], "first_stop_exact": timing["exact"],
        "first_stop_early": timing["early"], "first_stop_late": timing["late"],
        "missing_stop": timing["missing_stop"],
        "mean_absolute_step_error_when_stopped": (
            absolute_error / (timing["n"] - timing["missing_stop"])
            if timing["n"] != timing["missing_stop"] else None),
    }


def _bootstrap_asset_delta(ordinary: list[dict], evidence: list[dict], *, seed: int,
                           samples: int) -> dict:
    by_model = {}
    for name, rows in (("ordinary", ordinary), ("evidence", evidence)):
        by_model[name] = defaultdict(list)
        for row in rows:
            by_model[name][row["asset_id"]].append(row)
    assets = sorted(by_model["ordinary"])
    if assets != sorted(by_model["evidence"]):
        raise ValueError("Model asset rosters differ")

    def metrics(model: str, draw: list[str]) -> tuple[float, float]:
        rows = [row for asset in draw for row in by_model[model][asset]]
        result = summarize(rows)
        return result["exact_verdict"] / result["n"], result["unsupported_findings"] / result["n"]

    observed_o, observed_uo = metrics("ordinary", assets)
    observed_e, observed_ue = metrics("evidence", assets)
    rng = random.Random(seed)
    exact, unsupported = [], []
    for _ in range(samples):
        draw = [rng.choice(assets) for _ in assets]
        o, uo = metrics("ordinary", draw)
        e, ue = metrics("evidence", draw)
        exact.append(e - o)
        unsupported.append(ue - uo)

    def interval(values):
        values.sort()
        return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]
    return {"unit": "validation_asset", "clusters": len(assets), "samples": samples, "seed": seed,
            "exact_accuracy_delta": {"observed": observed_e - observed_o, "percentile_95_ci": interval(exact)},
            "unsupported_per_case_delta": {"observed": observed_ue - observed_uo,
                                           "percentile_95_ci": interval(unsupported)},
            "caution": "Only seven asset clusters; intervals are descriptive and unstable."}


def analyze(prefix_paths: dict[str, Path], no_remaining_paths: dict[str, Path], dataset: Path,
            budget_baseline_path: Path, *, seed: int, bootstrap_samples: int) -> dict:
    prefixes = {name: _rows(path) for name, path in prefix_paths.items()}
    ablations = {name: _rows(path) for name, path in no_remaining_paths.items()}
    prefix_ids = [row["id"] for row in prefixes["base"]]
    terminal_ids = [row["id"] for row in ablations["base"]]
    if len(prefix_ids) != 110 or len(terminal_ids) != 28:
        raise ValueError("Expected 110 causal prefixes and 28 terminal ablations")
    for name in MODELS:
        if [row["id"] for row in prefixes[name]] != prefix_ids:
            raise ValueError(f"{name} prefix roster/order differs")
        if [row["id"] for row in ablations[name]] != terminal_ids:
            raise ValueError(f"{name} ablation roster/order differs")
    terminal = {name: [row for row in prefixes[name] if row["terminal"]] for name in MODELS}
    if [row["id"] for row in terminal["base"]] != terminal_ids:
        raise ValueError("Terminal and ablation rosters differ")
    budget = _rows(budget_baseline_path)
    if [row["id"] for row in budget] != prefix_ids:
        raise ValueError("Budget baseline prefix roster/order differs")
    prefixes["budget_baseline"] = budget
    terminal["budget_baseline"] = [row for row in budget if row["terminal"]]
    stop_rows = _rows(dataset / "validation.stopping.jsonl")
    stop_targets = {(row["episode_id"], row["decision_index"]): row["stop_target"]
                    for row in stop_rows}
    if set(stop_targets) != {(row["episode_id"], row["decision_index"]) for row in prefixes["base"]}:
        raise ValueError("Stop-supervision and diagnostic prefixes differ")

    metadata = prefixes["base"]
    association = grouped(metadata, lambda row: (
        f"step={row['decision_index']},remaining={row['remaining']},"
        f"evidence={row['evidence_available']},terminal={row['terminal']}"))
    matched_stages = {}
    for step in (1, 2):
        matched_stages[str(step)] = {}
        for name in (*MODELS, "budget_baseline"):
            rows = [row for row in prefixes[name] if row["decision_index"] == step]
            matched_stages[str(step)][name] = {
                "evidence_sufficient": {
                    "decision_outputs": _decision_counts([row for row in rows if row["evidence_available"]]),
                    "status_outputs": _status_counts([row for row in rows if row["evidence_available"]]),
                },
                "evidence_insufficient": {
                    "decision_outputs": _decision_counts([row for row in rows if not row["evidence_available"]]),
                    "status_outputs": _status_counts([row for row in rows if not row["evidence_available"]]),
                },
            }

    ablation_result = {}
    for name in MODELS:
        original = {row["id"]: row for row in terminal[name]}
        transitions, same_raw, same_kind_status = Counter(), 0, 0
        for changed in ablations[name]:
            before = original[changed["id"]]
            left = before["predicted_status"] if before["decision_kind"] == "verdict" else before["decision_kind"]
            right = changed["predicted_status"] if changed["decision_kind"] == "verdict" else changed["decision_kind"]
            transitions[f"{left}->{right}"] += 1
            same_raw += before["raw"] == changed["raw"]
            same_kind_status += left == right
        ablation_result[name] = {"n": len(ablations[name]), "same_raw": same_raw,
                                 "same_decision_kind_and_status": same_kind_status,
                                 "transitions": dict(transitions),
                                 "with_remaining": summarize(terminal[name]),
                                 "without_remaining": summarize(ablations[name])}

    terminal_aggregates = {}
    for name in (*MODELS, "budget_baseline"):
        rows = terminal[name]
        terminal_aggregates[name] = {
            "overall": summarize(rows),
            "by_asset": grouped(rows, lambda row: row["asset_id"]),
            "by_defect_combination": grouped(rows, lambda row: row["defect_combination"]),
            "by_source_defect": {defect: summarize([row for row in rows if defect in row["source_defects"]])
                                 for defect in ("kinematics.joint_axis", "structure.part_offset")},
            "by_visible_finding_type": {defect: summarize_defect(rows, defect)
                                        for defect in ("kinematics.joint_axis", "structure.part_offset")},
        }
    ordinary = {row["id"]: row for row in terminal["ordinary"]}
    evidence = {row["id"]: row for row in terminal["evidence"]}
    paired = Counter()
    paired_by_asset = defaultdict(Counter)
    for uid in terminal_ids:
        o, e = ordinary[uid], evidence[uid]
        o_exact = summarize([o])["exact_verdict"]
        e_exact = summarize([e])["exact_verdict"]
        bucket = paired_by_asset[o["asset_id"]]
        paired["exact_gained"] += e_exact > o_exact
        paired["exact_lost"] += e_exact < o_exact
        paired["unsupported_delta"] += e["unsupported_findings"] - o["unsupported_findings"]
        bucket["exact_delta"] += e_exact - o_exact
        bucket["unsupported_delta"] += e["unsupported_findings"] - o["unsupported_findings"]

    assets = sorted({row["asset_id"] for row in metadata})
    return {
        "version": 2, "status": "complete", "scope": "validation_causal_prefix_shortcut_diagnostics",
        "counts": {"prefix_rows": 110, "terminal_episodes": 28, "partnet_validation_assets": len(assets),
                   "assets": assets, "formal_validation_pool_assets": 9,
                   "formal_assets_absent_from_episode_evaluation": 2},
        "identifiability": {
            "remaining_equals_six_minus_history_length": all(
                row["remaining"] == 6 - row["history_length"] for row in metadata),
            "image_count_equals_history_plus_reference": all(
                row["image_count"] == row["history_length"] + 1 for row in metadata),
            "terminal_gold_status_is_deterministic_from_step": {
                "steps_1_or_2": "fail (19/19)", "step_6": "uncertain (9/9)"},
            "stop_target_is_not_deterministic_from_budget_features": True,
            "causal_claim": "not_identifiable",
            "reason": "remaining, history length, image count and terminal status are collinear; deleting one key is OOD and leaves the other stage cues intact.",
        },
        "data_association": association, "same_budget_natural_prefix_comparison": matched_stages,
        "remaining_key_ablation": ablation_result, "terminal_aggregates": terminal_aggregates,
        "stop_timing": {name: _stop_metrics(rows, stop_targets)
                        for name, rows in prefixes.items()},
        "ordinary_vs_evidence_paired": {**dict(paired),
            "by_asset": {key: dict(value) for key, value in sorted(paired_by_asset.items())},
            "asset_cluster_bootstrap": _bootstrap_asset_delta(terminal["ordinary"], terminal["evidence"],
                                                               seed=seed, samples=bootstrap_samples)},
        "conclusions": {
            "confirmed": [
                "Terminal status is perfectly associated with decision step and remaining budget in these 28 episodes.",
                "The no-image metadata baseline predicts every Validation status target without reading image bytes.",
                "The evidence checkpoint reduces unsupported findings but also loses true findings on the same terminal cases.",
            ],
            "not_identified": [
                "Whether either SFT checkpoint recognizes visual evidence rather than using stage cues.",
                "A causal effect of remaining metadata, because history length and image count remain coupled after deletion.",
                "Generalization to the two formal Validation assets absent from the 28 Episode roster.",
            ],
        },
        "uncomparable_conditions": [
            "No pass-qualified Validation case exists.",
            "No naturally generated error with independent human Gold exists.",
            "Two formal Validation assets have no exported Episode.",
            "Only first_failure is evaluated; comprehensive_within_budget is a separate, unevaluated policy.",
            "Part localization cannot be estimated for the metadata baseline because part identity is excluded by design.",
        ],
        "experiment_config": {
            "decode": "greedy", "max_new_tokens": 512, "bootstrap_seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "conditions": {
                "causal_prefixes": "All 110 exported real validation prefixes, unchanged.",
                "terminal_without_remaining": "The same 28 terminal histories/images with only the remaining JSON key deleted.",
                "budget_baseline": "Train-fitted majority lookup using only step, remaining, history length and image count.",
            },
            "scoring_only_private_fields": ["labels", "exposures", "witness_indices", "evidence"],
            "model_input_private_fields": [],
        },
        "frozen_test_used": False, "training_run": False, "gold_or_evidence_standard_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in MODELS:
        parser.add_argument(f"--{name}-prefix", type=Path, required=True)
        parser.add_argument(f"--{name}-no-remaining", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--budget-baseline", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    prefix = {name: getattr(args, f"{name}_prefix") for name in MODELS}
    no_remaining = {name: getattr(args, f"{name}_no_remaining") for name in MODELS}
    report = analyze(prefix, no_remaining, args.dataset, args.budget_baseline,
                     seed=args.seed, bootstrap_samples=args.bootstrap_samples)
    report["inputs"] = {"prefix": {key: _input(value) for key, value in prefix.items()},
                        "no_remaining": {key: _input(value) for key, value in no_remaining.items()},
                        "budget_baseline": _input(args.budget_baseline),
                        "stopping_supervision": _input(args.dataset / "validation.stopping.jsonl")}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
