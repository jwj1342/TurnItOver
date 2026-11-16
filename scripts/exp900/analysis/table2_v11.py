"""Table 2 (tables/acquisition.tex) on three-tier-900-sft-v1.1-pre.1, following her definition (issue #10, 2026-11-16):
admitted validation cases; viewpoints = rest cells of Hard's views (Easy/Medium N/E), joints = every joint state of the
first view, full = the complete fixed sequence. Visibility = share of defective cases with an evidence target finding
under the condition (pixel-exposure proxy). Defect F1 = case-level macro-F1 over the five classes against the condition's
evidence target findings (a class without positives scores 0); invalid outputs count as no findings. SFT rows are
mean [min, max] over seeds. Every evaluation ran on L40S.
Usage: python scripts/exp900/analysis/table2_v11.py  -> prints and writes $TIO_RUNS/out/RESULTS-table2-v11.md
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

TIO = Path(os.environ.get("TIO_RUNS", Path(__file__).resolve().parents[3] / "tio-runs"))
OUT = TIO / "out"
EXPORTS = {"viewpoints": TIO / "data900/three-tier-900-cond-viewpoints-v11/episode_export",
           "joints": TIO / "data900/three-tier-900-cond-joints-v11/episode_export",
           "full": TIO / "data900/three-tier-900-sft-v1.1"}
TAXONOMY = ("kinematics.direction_flip", "kinematics.joint_anchor", "kinematics.joint_axis", "structure.part_offset",
            "structure.part_rotation")
TIERS = ("easy", "medium", "hard")


def targets(export: Path) -> dict:
    """episode id -> (tier, defect ids of the evidence target findings, clean); clean = no annotated label."""
    labels = {json.loads(l)["id"]: json.loads(l)["labels"] for l in (export / "annotations.private.jsonl").open()}
    out = {}
    for line in (export / "validation.episodes.jsonl").open():
        ep = json.loads(line)
        if ep.get("admission", "eligible") != "eligible":
            continue
        verdict = json.loads(ep["messages"][-1]["content"])["verdict"]
        out[ep["episode_id"]] = (ep["difficulty"], {f["defect_id"] for f in verdict["findings"]}, not labels[ep["episode_id"]])
    return out


def macro_f1(rows) -> float:
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
        tier, g, _ = gold[r["episode_id"]]
        row = (g, {f[0] for f in r.get("predicted_findings") or []})
        by[tier].append(row)
        by["all"].append(row)
    if len(by["all"]) != len(gold):
        raise ValueError(f"{pred}: {len(by['all'])} predictions for {len(gold)} cases")
    return {t: macro_f1(rows) for t, rows in by.items()}


def complete(pred: Path) -> bool:
    m = pred.with_name("metrics.json")
    return m.exists() and json.loads(m.read_text()).get("status") == "complete"


def runs() -> dict:
    found = defaultdict(list)
    for cond in ("viewpoints", "joints"):
        for pred in OUT.glob(f"cond11-{cond}-*/*/*-withheld/predictions.jsonl"):
            name = pred.parent.name
            sup = "base" if name.startswith("base") else name.split("-")[1]
            if complete(pred):
                found[cond, pred.parents[1].name, sup].append(pred)
    for pred in OUT.glob("level900v11-sft-*/*/seed*/*-diagnosis/eval-validation/predictions.jsonl"):
        if complete(pred):
            found["full", pred.parents[3].name, pred.parents[1].name.split("-")[0]].append(pred)
    for pred in OUT.glob("level900v11-base-*/*/base-validation/predictions.jsonl"):
        if complete(pred):
            found["full", pred.parents[1].name, "base"].append(pred)
    return found


def cell(values) -> str:
    if not values:
        return "--"
    mean = sum(values) / len(values)
    return f"{mean:.1f} [{min(values):.1f}, {max(values):.1f}]" if len(values) > 1 else f"{mean:.1f}"


def main() -> None:
    gold = {c: targets(e) for c, e in EXPORTS.items()}
    lines = ["# Table 2 on three-tier-900-sft-v1.1-pre.1 (admitted validation, L40S)", "",
             "Visibility: defective cases with an evidence target finding under the condition / defective cases in it.", "",
             "| Condition | Easy | Medium | Hard | All |", "|---|---:|---:|---:|---:|"]
    for cond in ("viewpoints", "joints", "full"):
        cells = []
        for tier in TIERS + ("all",):
            cases = [v for v in gold[cond].values() if (tier == "all" or v[0] == tier) and not v[2]]
            if cond == "viewpoints" and tier in ("easy", "medium"):
                cells.append("N/E")
                continue
            hit = sum(bool(v[1]) for v in cases)
            cells.append(f"{100 * hit / len(cases):.1f} ({hit}/{len(cases)})" if cases else "--")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "Defect macro-F1 against the condition's evidence targets (mean [min, max] over seeds):", "",
              "| Condition | Size | Model | runs | Easy | Medium | Hard | All |", "|---|---|---|---:|---:|---:|---:|---:|"]
    found = runs()
    for cond in ("viewpoints", "joints", "full"):
        for size in ("2B", "4B", "8B"):
            for sup in ("base", "ordinary", "evidence"):
                preds = found.get((cond, size, sup), [])
                if not preds:
                    continue
                scores = [score(p, gold[cond]) for p in preds]
                tiers = [("N/E" if cond == "viewpoints" and t in ("easy", "medium") else cell([s[t] for s in scores if t in s]))
                         for t in TIERS + ("all",)]
                lines.append(f"| {cond} | {size} | {sup} | {len(preds)} | " + " | ".join(tiers) + " |")
    (OUT / "RESULTS-table2-v11.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
