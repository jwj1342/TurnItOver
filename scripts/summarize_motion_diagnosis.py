"""Summarize new held-object and frozen A/B/C/D motion-diagnosis evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_modular_verifier import _read_jsonl
from scripts.score_motion_attribution import score, summary


def main(args):
    gold = {row["id"]: row for row in _read_jsonl(args.frozen / "gold.private.jsonl")}
    inputs = {(row["id"], row["condition"]): row for name in ("inputs.jsonl", "controls.inputs.jsonl")
              for row in _read_jsonl(args.frozen / name)}
    report = {"frozen_abcd": {}}
    scored = []
    for model in ("ordinary", "evidence"):
        rows = []
        for folder in (model, f"{model}-D"):
            for prediction in _read_jsonl(args.root / folder / "predictions.jsonl"):
                row = score(prediction, gold[prediction["id"]], inputs[prediction["id"], prediction["condition"]])
                row["model"] = model; rows.append(row); scored.append(row)
        report["frozen_abcd"][model] = {
            condition: summary([row for row in rows if row["condition"] == condition])
            for condition in ("A", "B", "C", "D")}
        report["frozen_abcd"][model]["per_object_B"] = {
            asset: summary([row for row in rows if row["condition"] == "B" and row["asset_id"] == asset])
            for asset in sorted({row["asset_id"] for row in rows})}
    report["held_object"] = {
        model: json.loads((args.held / f"partnet-qwen3vl2b-motion-{model}-eval-v1/metrics.json").read_text())
        for model in ("ordinary", "evidence")}
    (args.root / "scores.private.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scored))
    (args.root / "metrics.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--frozen",type=Path,required=True);p.add_argument("--held",type=Path,default=Path("output"))
    a=p.parse_args();a.root=a.root.resolve();a.frozen=a.frozen.resolve();a.held=a.held.resolve();main(a)
