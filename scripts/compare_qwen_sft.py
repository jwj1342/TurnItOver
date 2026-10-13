#!/usr/bin/env python3
"""Create a compact controlled comparison from two completed SFT evaluations."""
import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--sft", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise ValueError(f"Output must not exist: {args.out}")
base = json.loads(args.base.read_text())
sft = json.loads(args.sft.read_text())
if base.get("status") != "complete" or sft.get("status") != "complete":
    raise ValueError("Both evaluations must be complete")
for key in ("dataset_manifest_sha256", "validation_episodes_sha256", "max_new_tokens", "n"):
    if base.get(key) != sft.get(key):
        raise ValueError(f"Controlled comparison mismatch: {key}")
metrics = ("valid_output_rate", "stop_accuracy", "status_accuracy", "exact_verdict_accuracy")
report = {"version": 1, "status": "complete", "protocol": {
              key: base[key] for key in ("dataset_manifest_sha256", "validation_episodes_sha256",
                                         "max_new_tokens", "n")},
          "base": {key: base[key] for key in metrics},
          "sft": {key: sft[key] for key in metrics},
          "delta": {key: sft[key] - base[key] for key in metrics},
          "finding_micro": {"base": base["finding_micro"], "sft": sft["finding_micro"]},
          "limitations": sft["limitations"]}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
