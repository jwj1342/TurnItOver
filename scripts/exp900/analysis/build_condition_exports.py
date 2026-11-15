"""Derive observation-condition validation exports from the three-tier-900 episode export (all 120 validation cases).

  viewpoints: keep the rest-state cells of every view (what request_view alone reaches).
  joints:     keep every joint-state cell of the first view (what actuate_joint alone reaches).
The first reference image (image_order[0], first view at rest) belongs to both conditions and is always kept. History
steps outside the condition are dropped and the rest renumbered; the user prompt is rebuilt byte-exactly in her format
with remaining=0. The evidence-constrained target keeps each finding whose cited steps survive (renumbered); a case with
labels and no surviving finding gets the uncertain verdict of the withheld export. Easy and Medium have one view, so
"viewpoints" keeps only their rest cell. Recognition under each condition is scored offline against the annotated labels
by scripts/exp900/analysis/condition_f1.py.
Usage: python scripts/exp900/analysis/build_condition_exports.py viewpoints|joints
  -> $TIO_RUNS/data900/three-tier-900-cond-<condition>-v1/episode_export
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

TIO = Path(os.environ.get("TIO_RUNS", Path(__file__).resolve().parents[3] / "tio-runs"))
SRC = TIO / "data900/three-tier-900-sft-v1/episode_export"
NAME = re.compile(r"^\d+-(?:(?P<view>.+)_)?(?P<kind>reference|candidate)_(?P<state>rest|zero|mid|limit)\.png$")


def cell(image_ref: str) -> tuple[str, str]:
    m = NAME.match(image_ref.rsplit("/", 1)[-1])
    return m["view"] or "default", m["state"]


def main() -> None:
    condition = sys.argv[1]
    dst = TIO / f"data900/three-tier-900-cond-{condition}-v1/episode_export"
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    ann = {json.loads(l)["id"]: json.loads(l) for l in (SRC / "annotations.private.jsonl").open()}
    out, kept_cells = [], Counter()
    for line in (SRC / "validation.episodes.jsonl").open():
        ep = json.loads(line)
        content = ep["messages"][0]["content"]
        i = content.index('{"context"')
        prefix, payload = content[:i], json.loads(content[i:])
        assert prefix + json.dumps(payload) == content, ep["episode_id"]
        first_view = cell(payload["image_order"][0])[0]
        keep = (lambda c: c[1] == "rest") if condition == "viewpoints" else (lambda c: c[0] == first_view)
        history = [h for h in payload["history"] if keep(cell(h["image_ref"]))]
        renumber = {h["step"]: n for n, h in enumerate(history)}
        images = [payload["image_order"][0]] + [h["image_ref"] for h in history]
        target = json.loads(ep["messages"][-1]["content"])["verdict"]
        findings = [{**f, "evidence_steps": [renumber[s] for s in f["evidence_steps"] if s in renumber]} for f in target["findings"]]
        findings = [f for f in findings if f["evidence_steps"]]
        if findings or not target["findings"]:
            verdict = {**target, "findings": findings}
        else:
            verdict = {"status": "uncertain", "confidence": 0.0, "findings": [],
                       "summary": "The remaining observations do not expose a defect, but do not establish that the candidate is clean.",
                       "limitations": ["Observations outside this condition were not acquired."]}
        new_payload = {**payload, "history": [{**h, "step": renumber[h["step"]]} for h in history], "image_order": images, "remaining": 0}
        sha = dict(zip(ep["images"], ep["image_sha256"]))
        covered = ep["evidence"]["covered_label_indices"] if findings else []
        rec = {**ep, "images": images, "image_sha256": [sha[p] for p in images],
               "messages": [{"role": "user", "content": prefix + json.dumps(new_payload)},
                            {"role": "assistant", "content": json.dumps({"verdict": verdict})}],
               "evidence": {**ep["evidence"], "covered_label_indices": covered,
                            "uncovered_label_indices": [k for k in range(len(ann[ep["episode_id"]]["labels"])) if k not in covered],
                            "status": f"condition-{condition}"}}
        out.append(rec)
        kept_cells[ep["difficulty"]] += len({cell(p) for p in images})
        for p in images:
            (dst / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SRC / p, dst / p)
    (dst / "validation.episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    (dst / "annotations.private.jsonl").write_text("".join(json.dumps(ann[r["episode_id"]]) + "\n" for r in out))
    man = json.loads((SRC / "manifest.json").read_text())
    tiers = Counter(r["difficulty"] for r in out)
    man["counts"] = {"validation": {"difficulty": dict(tiers), "episodes": len(out), "stopping": 0},
                     "train": {"episodes": 0, "stopping": 0}, "test": {"episodes": 0, "stopping": 0}}
    man["dataset_version"] = f"three-tier-900-cond-{condition}-v1"
    man["derived_from"] = {"dataset_version": "three-tier-900-sft-v1",
                           "manifest_sha256": hashlib.sha256((SRC / "manifest.json").read_bytes()).hexdigest()}
    man["condition_rule"] = __doc__.split("\n\n")[0] + f" Condition: {condition}. Built by scripts/exp900/analysis/build_condition_exports.py."
    (dst / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    (dst / "PROVENANCE.json").write_text(json.dumps({"condition": condition, "cases": dict(tiers),
                                                     "cells_per_tier": dict(kept_cells)}, indent=2) + "\n")
    print(condition, dict(tiers), "cells", dict(kept_cells), "manifest sha256", hashlib.sha256((dst / "manifest.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
