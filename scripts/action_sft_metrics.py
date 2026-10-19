"""Summarize or merge disjoint policy runs with identical experimental inputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def summarize_rows(scored):
    groups, objects = defaultdict(list), defaultdict(list)
    for row in scored:
        groups[f"k{row['budget']}/{row['mode']}"].append(row)
        objects[row["asset_id"]].append(row)
    summary = {}
    for key, rows in groups.items():
        nonclean = [r for r in rows if r["labels"]]
        summary[key] = dict(episodes=len(rows), independent_objects=len({r["asset_id"] for r in rows}),
            defective_cases=len(nonclean), clean_cases=len(rows)-len(nonclean),
            valid_evidence_rate=sum(r["valid_evidence"] for r in rows)/len(rows),
            all_evidence_rate=sum(r["all_evidence"] for r in rows)/len(rows),
            nonclean_any_evidence_rate=sum(r["valid_evidence"] for r in nonclean)/len(nonclean) if nonclean else None,
            nonclean_all_evidence_rate=sum(r["all_evidence"] for r in nonclean)/len(nonclean) if nonclean else None,
            exposure_recall=sum(r["exposed"] for r in rows)/sum(r["labels"] for r in rows) if nonclean else None,
            diagnosis_accuracy=sum(r["diagnostic_exact"] for r in rows)/len(rows),
            mean_observation_cost=sum(r["spent"] for r in rows)/len(rows),
            legal_action_rate=sum(r["legal_actions"] for r in rows)/max(1, sum(r["action_attempts"] for r in rows)))
    paired = []
    for asset, rows in objects.items():
        for budget in sorted({row["budget"] for row in rows}):
            subset = [r for r in rows if r["budget"] == budget]
            averages = {}
            for mode in {r["mode"] for r in subset}:
                selected = [r for r in subset if r["mode"] == mode]
                averages[mode] = {metric: sum(r[metric] for r in selected)/len(selected)
                    for metric in ("valid_evidence", "all_evidence", "diagnostic_exact", "spent", "exposed")}
            for baseline in ("fixed", "random", "zero_shot"):
                if "action_sft" in averages and baseline in averages:
                    paired.append(dict(asset_id=asset, budget=budget, baseline=baseline,
                        delta_action_sft_minus_baseline={metric: averages["action_sft"][metric]-averages[baseline][metric]
                                                        for metric in averages[baseline]}))
    return summary, paired, dict(objects)


def merge(runs, output):
    if output.exists():
        raise ValueError("Fresh metrics output required")
    reports = [json.loads((root/"metrics.json").read_text()) for root in runs]
    configs = [json.loads((root/"run_config.json").read_text()) for root in runs]
    keys = ("source_sha256", "reference_sha256", "budgets", "action_tokens", "final_tokens")
    if any(report["status"] != "complete" or not report["score_eligible"] for report in reports):
        raise ValueError("Only completed model-scored runs can be merged")
    if any(any(config[key] != configs[0][key] for key in keys) for config in configs[1:]):
        raise ValueError("Runs have different public input/code/config hashes")
    rows, seen = [], set()
    for root, report in zip(runs, reports):
        for row in report["episodes"]:
            key = row["id"], row["mode"], row["budget"]
            if key in seen:
                raise ValueError("Duplicate episode across runs")
            seen.add(key)
            rows.append(dict(row, run=str(root.resolve())))
    cases = defaultdict(set)
    for uid, mode, budget in seen:
        cases[mode].add((uid, budget))
    if set(cases) != {"zero_shot", "action_sft", "fixed", "random"} or any(v != next(iter(cases.values())) for v in cases.values()):
        raise ValueError("Four policies must have identical Validation IDs and budgets")
    summary, paired, objects = summarize_rows(rows)
    report = dict(status="complete", score_eligible=True, frozen_test_used=False, summary=summary,
                  per_object_paired_deltas=paired, per_object=objects, episodes=rows,
                  offline_privileged_oracle=reports[0]["offline_privileged_oracle"],
                  runs_sha256={str(root.resolve()): hashlib.sha256((root/"metrics.json").read_bytes()).hexdigest() for root in runs})
    output.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    merge(args.runs, args.out)
