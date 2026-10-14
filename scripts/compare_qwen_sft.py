#!/usr/bin/env python3
"""Create a compact controlled comparison from completed SFT evaluations."""
import argparse
import hashlib
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--sft", type=Path)
parser.add_argument("--ordinary", type=Path)
parser.add_argument("--evidence", type=Path)
parser.add_argument("--base-stopping", type=Path)
parser.add_argument("--ordinary-stopping", type=Path)
parser.add_argument("--evidence-stopping", type=Path)
parser.add_argument("--budget-baseline", type=Path)
parser.add_argument("--ordinary-train", type=Path)
parser.add_argument("--evidence-train", type=Path)
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

round2_paths = (args.base_stopping, args.ordinary_stopping, args.evidence_stopping,
                args.budget_baseline, args.ordinary_train, args.evidence_train)
if any(round2_paths):
    if not all(round2_paths) or "ordinary" not in runs or "evidence" not in runs:
        raise ValueError("Round-2 audit requires all stopping, budget, training, and diagnosis inputs")
    stopping_paths = {name: getattr(args, f"{name}_stopping")
                      for name in ("base", "ordinary", "evidence")}
    stopping = {name: json.loads(path.read_text()) for name, path in stopping_paths.items()}
    budget = json.loads(args.budget_baseline.read_text())
    training_paths = {"ordinary": args.ordinary_train, "evidence": args.evidence_train}
    training = {name: json.loads(path.read_text()) for name, path in training_paths.items()}
    if any(value.get("status") != "complete" for value in (*stopping.values(), budget, *training.values())):
        raise ValueError("Round-2 inputs must be complete")
    controlled_keys = ("dataset_manifest_sha256", "train_episodes_sha256",
                       "train_stopping_sha256", "base_model_config_sha256", "seed", "epochs",
                       "learning_rate", "lora", "examples", "steps")
    mismatches = {key: [training[name].get(key) for name in ("ordinary", "evidence")]
                  for key in controlled_keys
                  if training["ordinary"].get(key) != training["evidence"].get(key)}
    if mismatches:
        raise ValueError(f"Training controls differ: {mismatches}")
    if training["ordinary"]["objectives"]["counts"] != training["evidence"]["objectives"]["counts"]:
        raise ValueError("Training objective counts differ")
    if training["ordinary"]["objectives"]["loss_weights"] != training["evidence"]["objectives"]["loss_weights"]:
        raise ValueError("Training objective weights differ")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def stop_rate(summary):
        confusion = summary["decision_confusion"]
        stops = sum(value for key, value in confusion.items() if key.endswith("->stop"))
        return stops / summary["n"]

    stage = {}
    for name in ("ordinary", "evidence"):
        groups = stopping[name]["same_stage_evidence"]
        stage[name] = {}
        for step in (1, 2):
            absent = groups[f"step={step},evidence=false"]
            present = groups[f"step={step},evidence=true"]
            stage[name][str(step)] = {
                "evidence_absent_stop_rate": stop_rate(absent),
                "evidence_present_stop_rate": stop_rate(present),
                "desired_direction": stop_rate(present) - stop_rate(absent),
            }
    ordinary_diag, evidence_diag = runs["ordinary"], runs["evidence"]
    report.update({
        "version": 3,
        "round2_training": {
            "controlled_equal": True,
            "shared": {key: training["ordinary"][key] for key in controlled_keys},
            "objective_counts": training["ordinary"]["objectives"]["counts"],
            "objective_loss_weights": training["ordinary"]["objectives"]["loss_weights"],
            "supervision_sha256": {name: training[name]["objectives"]["supervision_sha256"]
                                   for name in training},
            "unavoidable_target_difference": (
                "The 110 ordinary terminal diagnoses include all annotated findings; the evidence "
                "targets include only findings supported by the visible causal prefix. All 429 "
                "continue/stop targets and every public input are shared."),
            "nondeterminism_boundary": (
                "Seeds and ordering are fixed, but CUDA CuBLAS/Flash Attention emitted warn-only "
                "nondeterminism warnings; checkpoints are not claimed bitwise reproducible."),
        },
        "round2_diagnostic_delta_evidence_minus_ordinary": {
            "exact_verdict_accuracy": evidence_diag["exact_verdict_accuracy"] - ordinary_diag["exact_verdict_accuracy"],
            "finding_f1": evidence_diag["finding_micro"]["f1"] - ordinary_diag["finding_micro"]["f1"],
            "defect_type_f1": evidence_diag["defect_type_micro"]["f1"] - ordinary_diag["defect_type_micro"]["f1"],
            "part_localization_f1": evidence_diag["part_localization_micro"]["f1"] - ordinary_diag["part_localization_micro"]["f1"],
            "unsupported_findings": (evidence_diag["evidence_consistency"]["unsupported_findings"]
                                     - ordinary_diag["evidence_consistency"]["unsupported_findings"]),
            "finding_tp": evidence_diag["finding_micro"]["tp"] - ordinary_diag["finding_micro"]["tp"],
            "finding_fp": evidence_diag["finding_micro"]["fp"] - ordinary_diag["finding_micro"]["fp"],
            "finding_fn": evidence_diag["finding_micro"]["fn"] - ordinary_diag["finding_micro"]["fn"],
        },
        "round2_stopping": {name: value["overall"] for name, value in stopping.items()},
        "same_stage_evidence": stage,
        "no_image_budget_baseline": budget,
        "coverage": {
            "validation_prefixes": 110, "terminal_episodes": 28,
            "validation_assets": len(runs["ordinary"]["per_asset"]),
            "asset_ids": sorted(runs["ordinary"]["per_asset"]),
        },
        "conclusions": {
            "evidence_sft_more_accurate_visual_use": False,
            "prefix_stopping_truly_improved": False,
            "supported": (
                "Evidence SFT reduces unsupported terminal findings by four and improves defect-type "
                "F1, without changing exact finding or localization F1."),
            "counterevidence": (
                "Evidence SFT has lower stop accuracy and more premature stops than ordinary SFT; "
                "both fail all step-1 evidence-absent prefixes, and the no-image baseline is stronger."),
            "missing_control": (
                "The held-out set lacks stage-matched image counterfactual pairs that preserve legal "
                "history while changing only evidence content, as well as multiple training seeds."),
        },
        "artifact_sha256": {
            "diagnosis": {name: {"metrics": sha(path),
                                  "predictions": sha(path.with_name("predictions.jsonl"))}
                          for name, path in paths.items()},
            "stopping": {name: {"metrics": sha(path),
                                 "predictions": sha(path.with_name("predictions.jsonl"))}
                         for name, path in stopping_paths.items()},
            "budget_baseline": {"metrics": sha(args.budget_baseline),
                                "predictions": sha(args.budget_baseline.with_name("predictions.jsonl"))},
            "training": {name: {"run": sha(path),
                                "loss": sha(path.with_name("loss.jsonl")),
                                "adapter": sha(path.parent / "checkpoint/adapter_model.safetensors")}
                         for name, path in training_paths.items()},
        },
        "frozen_part2_test_used": False,
        "api_collection_used": False,
        "privileged_oracle_actions_trained": False,
    })
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
