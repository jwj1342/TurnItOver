"""Case-level exposure of the three-tier-900 validation (or test) split under three observation conditions.

A defective case is exposed under a condition when at least one of the condition's candidate cells is cited by the
evidence-constrained target (pixel exposure, as in the export). Cells are (view, state) pairs parsed from the image names.
  Viewpoints only: the rest state in every view.      Joints only: every joint state in the first view.
  Complete fixed scan: every cell.
Easy and Medium have one view, so for them "viewpoints only" is the single rest cell. Case-level exposure is unaffected by
the export crediting a pixel difference to the first label only (any exposing cell exposes the case).
Usage: python scripts/exp900/analysis/exposure_by_condition.py [split]   -> writes $TIO_RUNS/out/RESULTS-exposure-conditions-<split>.md
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

TIO = Path(os.environ.get("TIO_RUNS", Path(__file__).resolve().parents[3] / "tio-runs"))
EXPORT = TIO / "data900/three-tier-900-sft-v1/episode_export"
NAME = re.compile(r"^\d+-(?:(?P<view>.+)_)?(?P<kind>reference|candidate)_(?P<state>rest|zero|mid|limit)\.png$")
TIERS = ("easy", "medium", "hard")


def cell(image_ref: str) -> tuple[str, str, str]:
    m = NAME.match(image_ref.rsplit("/", 1)[-1])
    return m["view"] or "default", m["kind"], m["state"]


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    labels = {json.loads(l)["id"]: json.loads(l)["labels"] for l in (EXPORT / "annotations.private.jsonl").open()}
    counts = defaultdict(lambda: defaultdict(int))
    for line in (EXPORT / f"{split}.episodes.jsonl").open():
        ep = json.loads(line)
        if not labels[ep["episode_id"]]:
            continue
        payload = json.loads(ep["messages"][0]["content"].rsplit("\n", 1)[-1])
        images = payload["image_order"]
        step_image = {h["step"]: h["image_ref"] for h in payload["history"]}
        target = json.loads(ep["messages"][1]["content"])["verdict"]
        exposing = {cell(step_image[s])[::2] for f in target["findings"] for s in f["evidence_steps"]
                    if cell(step_image[s])[1] == "candidate"}
        cells = {cell(i)[::2] for i in images if cell(i)[1] == "candidate"}
        first_view = cell(images[0])[0]
        conditions = {"Viewpoints only": {c for c in cells if c[1] == "rest"},
                      "Joints only": {c for c in cells if c[0] == first_view},
                      "Complete fixed scan": cells}
        tier = ep["difficulty"]
        for name, shown in conditions.items():
            for key in (tier, "all"):
                counts[name][key, "n"] += 1
                counts[name][key, "exposed"] += bool(exposing & shown)
                counts[name][key, "cells"] += len(shown)
    out = [f"# Case-level exposure by observation condition, {split} split (defective cases)", "",
           "| Observation condition | Easy | Medium | Hard | All | cells per case (E/M/H) |", "|---|---:|---:|---:|---:|---|"]
    for name in ("Viewpoints only", "Joints only", "Complete fixed scan"):
        c = counts[name]
        pct = [f"{100 * c[k, 'exposed'] / c[k, 'n']:.1f} ({c[k, 'exposed']}/{c[k, 'n']})" for k in (*TIERS, "all")]
        per = "/".join(f"{c[t, 'cells'] / c[t, 'n']:.0f}" for t in TIERS)
        out.append(f"| {name} | " + " | ".join(pct) + f" | {per} |")
    path = TIO / f"out/RESULTS-exposure-conditions-{split}.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
