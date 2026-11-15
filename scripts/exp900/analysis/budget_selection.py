"""Observation budgets K for Fixed, Random and Active-oracle cell selection (tables/oracle_selection.tex), as a proxy
computed from the three-tier-900 pixel-exposure records (no model calls).

Action grid: the candidate cells (view, state) of each defective case, in fixed-scan order (order in image_order).
A cell exposes a label when the evidence-constrained target cites it for that label. That record credits a pixel
difference to the first label only, so label-level recall (Structure, Kinematics, All) uses single-defect cases only;
"Any case" (at least one exposing cell among the first K) and "First exposure" (1-based position of the first exposing
cell, cases exposed somewhere in the grid) use every defective case.
Policies: Fixed = scan order; Random = uniform shuffle, expectation over 100 seeds; Active oracle = greedy, each step takes
the cell exposing the most not-yet-exposed labels (scan order breaks ties), i.e. an upper bound with exposure known.
95% intervals: 1000 bootstrap resamples of source objects. Randomness only through numpy Generators with fixed seeds.
Usage: python scripts/exp900/analysis/budget_selection.py [test|validation]  -> $TIO_RUNS/out/RESULTS-budget-selection-<split>.md
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TIO = Path(os.environ.get("TIO_RUNS", Path(__file__).resolve().parents[3] / "tio-runs"))
EXPORT = TIO / "data900/three-tier-900-sft-v1/episode_export"
NAME = re.compile(r"^\d+-(?:(?P<view>.+)_)?(?P<kind>reference|candidate)_(?P<state>rest|zero|mid|limit)\.png$")
BUDGETS = (1, 2, 4, 6)
SEEDS = 100


def cell(image_ref: str) -> tuple[str, str, str]:
    m = NAME.match(image_ref.rsplit("/", 1)[-1])
    return m["view"] or "default", m["kind"], m["state"]


def load(split: str) -> list[dict]:
    ann = {json.loads(l)["id"]: json.loads(l) for l in (EXPORT / "annotations.private.jsonl").open()}
    cases = []
    for line in (EXPORT / f"{split}.episodes.jsonl").open():
        ep = json.loads(line)
        labels = [x["defect_id"] for x in ann[ep["episode_id"]]["labels"]]
        if not labels:
            continue
        payload = json.loads(ep["messages"][0]["content"].rsplit("\n", 1)[-1])
        step_image = {h["step"]: h["image_ref"] for h in payload["history"]}
        grid = [cell(i)[::2] for i in payload["image_order"] if cell(i)[1] == "candidate"]
        exposes = defaultdict(set)          # cell -> label indices
        for f in json.loads(ep["messages"][1]["content"])["verdict"]["findings"]:
            for s in f["evidence_steps"]:
                view, kind, state = cell(step_image[s])
                if kind == "candidate":
                    exposes[view, state].add(labels.index(f["defect_id"]))
        cases.append({"object": ep["data_identity"]["source_asset_id"], "labels": labels, "grid": grid, "exposes": exposes})
    return cases


def oracle_order(case: dict) -> list:
    left, seen, order = list(case["grid"]), set(), []
    while left:
        best = max(left, key=lambda c: len(case["exposes"][c] - seen))   # max keeps the first (scan order) on ties
        order.append(best); left.remove(best); seen |= case["exposes"][best]
    return order


def case_metrics(case: dict, order: list) -> dict:
    out = {}
    first = next((i + 1 for i, c in enumerate(order) if case["exposes"][c]), None)
    for k in BUDGETS:
        shown = set().union(*(case["exposes"][c] for c in order[:k])) if order[:k] else set()
        out[k] = {"any": float(bool(shown)), "label": {i: float(i in shown) for i in range(len(case["labels"]))}}
    out["first"] = first
    return out


def per_case(cases: list[dict], policy: str) -> list[dict]:
    rows = []
    for n, case in enumerate(cases):
        if policy == "fixed":
            orders = [case["grid"]]
        elif policy == "oracle":
            orders = [oracle_order(case)]
        else:
            rng = np.random.default_rng(20261115 + n)
            orders = [[case["grid"][i] for i in rng.permutation(len(case["grid"]))] for _ in range(SEEDS)]
        ms = [case_metrics(case, o) for o in orders]
        row = {"object": case["object"], "single": len(case["labels"]) == 1, "kind": case["labels"][0].split(".")[0]}
        for k in BUDGETS:
            row[k, "any"] = np.mean([m[k]["any"] for m in ms])
            row[k, "label"] = np.mean([m[k]["label"][0] for m in ms])
        firsts = [m["first"] for m in ms if m["first"] is not None]
        row["first"] = np.mean(firsts) if firsts else None
        rows.append(row)
    return rows


def stat(rows: list[dict], value, keep, rng) -> str:
    sel = [r for r in rows if keep(r) and value(r) is not None]
    if not sel:
        return "N/E"
    objects = sorted({r["object"] for r in sel})
    by_obj = {o: [value(r) for r in sel if r["object"] == o] for o in objects}
    point = np.mean([v for vs in by_obj.values() for v in vs])
    boots = []
    for _ in range(1000):
        pick = rng.choice(len(objects), len(objects))
        vals = [v for i in pick for v in by_obj[objects[i]]]
        boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    scale = 1 if value is FIRST else 100
    return f"{scale * point:.1f} [{scale * lo:.1f}, {scale * hi:.1f}]" if scale == 100 else f"{point:.2f} [{lo:.2f}, {hi:.2f}]"


def FIRST(r):
    return r["first"]


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    cases = load(split)
    rng = np.random.default_rng(7)
    lines = [f"# Budgeted cell selection, {split} split ({len(cases)} defective cases, "
             f"{sum(len(c['labels']) == 1 for c in cases)} single-defect), proxy from pixel-exposure records", "",
             "| K | Policy | Structure recall | Kinematics recall | All recall | Any case | First exposure |",
             "|---|---|---|---|---|---|---|"]
    latex = []
    for k in BUDGETS:
        for policy, name in (("fixed", "Fixed"), ("random", "Random"), ("oracle", "Active oracle")):
            rows = per_case(cases, policy)
            cells = [stat(rows, lambda r: r[k, "label"], lambda r: r["single"] and r["kind"] == "structure", rng),
                     stat(rows, lambda r: r[k, "label"], lambda r: r["single"] and r["kind"] == "kinematics", rng),
                     stat(rows, lambda r: r[k, "label"], lambda r: r["single"], rng),
                     stat(rows, lambda r: r[k, "any"], lambda r: True, rng),
                     stat(rows, FIRST, lambda r: True, rng)]
            lines.append(f"| {k} | {name} | " + " | ".join(cells) + " |")
            latex.append(f"{k} & {name} & " + " & ".join(cells) + r" \\")
    lines += ["", "LaTeX rows (Budget & Policy & Structure & Kinematics & All & Any case & First exposure):", "```", *latex, "```",
              "", "First exposure does not depend on K (position in the policy's order, cases exposed anywhere in the grid)."]
    path = TIO / f"out/RESULTS-budget-selection-{split}.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
