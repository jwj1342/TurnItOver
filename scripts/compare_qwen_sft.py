#!/usr/bin/env python3
"""Create a compact controlled comparison from completed SFT evaluations."""
import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--sft", type=Path)
parser.add_argument("--ordinary", type=Path)
parser.add_argument("--evidence", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise ValueError(f"Output must not exist: {args.out}")
paths = {"base": args.base}
if args.sft:
    if args.ordinary or args.evidence:
        raise ValueError("Use either --sft or the --ordinary/--evidence pair")
    paths["sft"] = args.sft
elif args.ordinary and args.evidence:
    paths.update(ordinary=args.ordinary, evidence=args.evidence)
else:
    raise ValueError("Expected --sft or both --ordinary and --evidence")
runs = {name: json.loads(path.read_text()) for name, path in paths.items()}
if any(run.get("status") != "complete" for run in runs.values()):
    raise ValueError("All evaluations must be complete")
for name, run in runs.items():
    for key in ("dataset_manifest_sha256", "validation_episodes_sha256", "max_new_tokens", "n"):
        if run.get(key) != runs["base"].get(key):
            raise ValueError(f"Controlled comparison mismatch for {name}: {key}")
metrics = ("valid_output_rate", "status_accuracy", "exact_verdict_accuracy")
limitations = [row for row in list(runs.values())[-1]["limitations"]
               if "stop accuracy" not in row.lower()]
limitations.append("Terminal valid_output_rate measures legal output only; stop timing is evaluated separately.")
report = {"version": 2, "status": "complete", "protocol": {
              key: runs["base"][key] for key in ("dataset_manifest_sha256", "validation_episodes_sha256",
                                                  "max_new_tokens", "n")},
          "runs": {name: {**{key: run[key] for key in metrics},
                           "model": run.get("model"), "adapter": run.get("adapter"),
                           "status_classification": run.get("status_classification"),
                           "false_fail_rate": run.get("false_fail_rate"),
                           "fail_false_positive_rate": run.get("fail_false_positive_rate"),
                           "false_pass_rate": run.get("false_pass_rate"),
                           "finding_micro": run["finding_micro"],
                           "defect_type_micro": run.get("defect_type_micro"),
                           "part_localization_micro": run.get("part_localization_micro"),
                           "finding_per_defect": run.get("finding_per_defect"),
                           "output_validity": run.get("output_validity") or {
                               "evaluation_point": "forced_terminal_prefix_only",
                               "legal_final_verdicts": round(run["valid_output_rate"] * run["n"]),
                           },
                           "expected_statuses": run.get("expected_statuses"),
                           "predicted_statuses": run.get("predicted_statuses"),
                           "per_asset": run.get("per_asset"),
                           "evidence_consistency": run.get("evidence_consistency")}
                   for name, run in runs.items()},
          "delta_vs_base": {name: {key: run[key] - runs["base"][key] for key in metrics}
                            for name, run in runs.items() if name != "base"},
          "limitations": limitations}
if "sft" in runs:
    report.update(base=report["runs"]["base"], sft=report["runs"]["sft"],
                  delta=report["delta_vs_base"]["sft"],
                  finding_micro={"base": runs["base"]["finding_micro"],
                                 "sft": runs["sft"]["finding_micro"]})
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
