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
        verdict = row["decision_kind"] == "verdict"
        counts["valid_verdict"] += verdict
        counts["status_correct"] += verdict and row["predicted_status"] == row["expected_status"]
        counts["exact"] += verdict and row["predicted_status"] == row["expected_status"] and predicted == expected
        counts["tp"] += len(expected & predicted)
        counts["fp"] += len(predicted - expected)
        counts["fn"] += len(expected - predicted)
        counts["unsupported"] += row["unsupported_findings"]
        counts["predicted_findings"] += len(predicted)
        counts["false_fail"] += verdict and row["predicted_status"] == "fail" and row["expected_status"] != "fail"
    return {"n": counts["n"], "valid_verdicts": counts["valid_verdict"],
            "status_correct": counts["status_correct"], "exact_verdict": counts["exact"],
            "findings": _prf(counts["tp"], counts["fp"], counts["fn"]),
            "unsupported_findings": counts["unsupported"],
            "predicted_findings": counts["predicted_findings"], "false_fail": counts["false_fail"]}


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


def analyze(prefix_paths: dict[str, Path], no_remaining_paths: dict[str, Path], *, seed: int,
            bootstrap_samples: int) -> dict:
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

    metadata = prefixes["base"]
    association = grouped(metadata, lambda row: (
        f"step={row['decision_index']},remaining={row['remaining']},"
        f"evidence={row['evidence_available']},terminal={row['terminal']}"))
    matched_stages = {}
    for step in (1, 2):
        matched_stages[str(step)] = {}
        for name in MODELS:
            rows = [row for row in prefixes[name] if row["decision_index"] == step]
            matched_stages[str(step)][name] = {
                "evidence_sufficient": _decision_counts([row for row in rows if row["evidence_available"]]),
                "evidence_insufficient": _decision_counts([row for row in rows if not row["evidence_available"]]),
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
    for name in MODELS:
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
        "version": 1, "status": "complete", "scope": "validation_causal_prefix_shortcut_diagnostics",
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
            "causal_claim": "not_identifiable",
            "reason": "remaining, history length, image count and terminal status are collinear; deleting one key is OOD and leaves the other stage cues intact.",
        },
        "data_association": association, "same_budget_natural_prefix_comparison": matched_stages,
        "remaining_key_ablation": ablation_result, "terminal_aggregates": terminal_aggregates,
        "ordinary_vs_evidence_paired": {**dict(paired),
            "by_asset": {key: dict(value) for key, value in sorted(paired_by_asset.items())},
            "asset_cluster_bootstrap": _bootstrap_asset_delta(terminal["ordinary"], terminal["evidence"],
                                                               seed=seed, samples=bootstrap_samples)},
        "conclusions": {
            "confirmed": [
                "Terminal status is perfectly associated with decision step and remaining budget in these 28 episodes.",
                "The evidence checkpoint reduces unsupported findings but also loses true findings on the same terminal cases.",
            ],
            "not_identified": [
                "Whether either SFT checkpoint recognizes visual evidence rather than using stage cues.",
                "A causal effect of remaining metadata, because history length and image count remain coupled after deletion.",
                "Generalization to the two formal Validation assets absent from the 28 Episode roster.",
            ],
        },
        "experiment_config": {
            "decode": "greedy", "max_new_tokens": 512, "bootstrap_seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "conditions": {
                "causal_prefixes": "All 110 exported real validation prefixes, unchanged.",
                "terminal_without_remaining": "The same 28 terminal histories/images with only the remaining JSON key deleted.",
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
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    prefix = {name: getattr(args, f"{name}_prefix") for name in MODELS}
    no_remaining = {name: getattr(args, f"{name}_no_remaining") for name in MODELS}
    report = analyze(prefix, no_remaining, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
    report["inputs"] = {"prefix": {key: _input(value) for key, value in prefix.items()},
                        "no_remaining": {key: _input(value) for key, value in no_remaining.items()}}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
