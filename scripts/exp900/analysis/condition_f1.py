"""Defect F1 under observation conditions (tables/acquisition.tex), scored offline against the annotated labels.

Rows: viewpoints only and joints only (runs cond-<condition>-<size>-<job>, from build_condition_exports.py) and the
complete fixed scan (the L40S re-evaluations level900-eval-<size>-<job> of the full validation export), so every row is
the same hardware. Per tier: case-level macro-F1 over the five taxonomy labels, a label absent from the tier scoring 0
as in her scorer, but against all annotated defects of each case (a condition that hides a defect costs recall).
Usage: python scripts/exp900/analysis/condition_f1.py  -> prints and writes $TIO_RUNS/out/RESULTS-condition-f1.md
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

TIO = Path(os.environ.get("TIO_RUNS", Path(__file__).resolve().parents[3] / "tio-runs"))
OUT = TIO / "out"
EXPORT = TIO / "data900/three-tier-900-sft-v1/episode_export"
TAXONOMY = ("kinematics.direction_flip", "kinematics.joint_anchor", "kinematics.joint_axis", "structure.part_offset",
            "structure.part_rotation")


def macro_f1(rows: list[tuple[set, set]]) -> float:
    scores = []
    for label in TAXONOMY:
        tp = sum(label in g and label in p for g, p in rows)
        fp = sum(label in p and label not in g for g, p in rows)
        fn = sum(label in g and label not in p for g, p in rows)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return 100 * sum(scores) / len(scores)


def score(pred: Path, gold: dict) -> dict:
    by = defaultdict(list)
    for line in pred.open():
        r = json.loads(line)
        row = (gold[r["episode_id"]], {f[0] for f in r.get("predicted_findings") or []})
        by[r["difficulty"]].append(row)
        by["all"].append(row)
    return {t: macro_f1(rows) for t, rows in by.items()}


def runs() -> dict:
    found = defaultdict(list)
    for cond in ("viewpoints", "joints"):
        for pred in OUT.glob(f"cond-{cond}-*/*/seed*-*-withheld/predictions.jsonl"):
            seed, sup = pred.parent.name.split("-")[:2]
            found[cond, pred.parents[1].name, sup].append(pred)
    for pred in OUT.glob("level900-eval-*/*/seed*/*-diagnosis/eval-validation/predictions.jsonl"):
        found["complete", pred.parents[3].name, pred.parents[1].name.split("-")[0]].append(pred)
    return found


def main() -> None:
    gold = {json.loads(l)["id"]: {x["defect_id"] for x in json.loads(l)["labels"]} for l in (EXPORT / "annotations.private.jsonl").open()}
    found = runs()
    lines = ["# Defect F1 by observation condition (validation, L40S, against annotated labels; mean [min, max] over seeds)", "",
             "| Condition | Size | Supervision | runs | Easy | Medium | Hard | All |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for cond in ("viewpoints", "joints", "complete"):
        for size in ("2B", "8B"):
            for sup in ("ordinary", "evidence"):
                preds = found.get((cond, size, sup), [])
                if not preds:
                    continue
                scores = [score(p, gold) for p in preds]
                cells = []
                for t in ("easy", "medium", "hard", "all"):
                    v = [s[t] for s in scores]
                    cells.append(f"{sum(v) / len(v):.1f} [{min(v):.1f}, {max(v):.1f}]" if len(v) > 1 else f"{v[0]:.1f}")
                lines.append(f"| {cond} | {size} | {sup} | {len(preds)} | " + " | ".join(cells) + " |")
    (OUT / "RESULTS-condition-f1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
