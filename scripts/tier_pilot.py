#!/usr/bin/env python3
"""Assign pilot samples to difficulty tiers from the human review and the per-view visibility audit.

Rule (post-hoc, pilot only): a defect sample is Hard ("needs an extra view") iff the reviewer did not see it in one
glance AND some other view passing the pixel gate shows at least RATIO times the default view's difference.
Hard samples expect ``continue`` under the fixed four-image input and ``fail`` once extra views are allowed.
Clean controls inherit their object's tier. Unresolved reviews get no tier. Writes private/tiering.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.construction_pilot import OUT

RATIO = 2.0


def needs_extra_view(one_glance, default_view: str, per_view: dict, ratio: float = RATIO) -> bool:
    usable = {v: m["core_changed_fraction"] for v, m in per_view.items() if m["passes_gate"] and m["in_frame"] and v != default_view}
    default = per_view[default_view]["core_changed_fraction"]
    return one_glance is False and bool(usable) and max(usable.values()) >= ratio * default


def assign(bundle: Path) -> dict:
    gold = {json.loads(l)["case_id"]: json.loads(l) for l in (bundle / "private/gold.jsonl").read_text().splitlines()}
    reviews = {json.loads(l)["case_id"]: json.loads(l) for l in (bundle / "human_review.jsonl").read_text().splitlines()}
    visibility = {r["case_id"]: r for r in json.loads((bundle / "private/view_visibility.json").read_text())["rows"]}
    rows, object_tier = {}, {}
    for cid, g in gold.items():
        if g["kind"] == "clean":
            continue
        review, vis = reviews[cid], visibility[cid]
        if review["status"] != "human_review_complete":
            rows[cid] = {"tier": None, "reason": "unresolved_human_review"}
        elif needs_extra_view(review["human_judgment"]["one_glance"], vis["chosen_view"], vis["per_view"]):
            rows[cid] = {"tier": "hard", "reason": "needs_extra_view", "fixed_input_expected": "continue",
                         "with_extra_views_expected": "fail", "default_view": vis["chosen_view"],
                         "default_fraction": vis["chosen_fraction"], "views_passing_gate": vis["views_passing_gate"]}
        else:
            rows[cid] = {"tier": g["difficulty"], "reason": "visible_in_default_view",
                         "fixed_input_expected": "fail", "with_extra_views_expected": "fail"}
        object_tier[g["asset_id"]] = rows[cid]["tier"]
    for cid, g in gold.items():
        if g["kind"] == "clean":
            tier = object_tier.get(g["asset_id"])
            rows[cid] = {"tier": tier, "reason": "inherits_object_tier" if tier else "paired_defect_unresolved",
                         "fixed_input_expected": "pass", "with_extra_views_expected": "pass"}
    return {"rule": {"hard_if": "one_glance is False and another gate-passing view shows >= ratio x the default view",
                     "ratio": RATIO, "post_hoc": True}, "samples": dict(sorted(rows.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=OUT)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    result = assign(bundle)
    (bundle / "private/tiering.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(Counter((r["tier"], r["reason"]) for r in result["samples"].values()))


if __name__ == "__main__":
    main()
