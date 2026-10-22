#!/usr/bin/env python3
"""For every pilot defect sample, measure how visible the defect is in each of the 12 views.

Uses the same frozen normal framing, joint state and pixel gate (core changed >= 2.5%, largest region >= 750 px)
as construction_pilot. Read-only on the pilot bundle except for writing private/view_visibility.json.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from scripts.construction_pilot import OUT
from scripts.level0_expansion_search import inside_camera, render_states
from scripts.prepare_level0_axis_review import full_normal_framing
from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, image_metrics
from turnitover.assets.catalog import decode_spec
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

GATE_FRACTION, GATE_REGION = 0.025, 750


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=OUT)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    gold = [json.loads(l) for l in (bundle / "private/gold.jsonl").read_text().splitlines()]
    public = {json.loads(l)["case_id"]: json.loads(l) for l in (bundle / "review_manifest.jsonl").read_text().splitlines()}
    reviews = {json.loads(l)["case_id"]: json.loads(l) for l in (bundle / "human_review.jsonl").read_text().splitlines()}
    screening = {a["asset_id"]: a for a in json.loads((bundle / "private/screening.json").read_text())["assets"]
                 if a["status"] == "automatic_pass"}
    cfg = load_generate_config(DEFAULT_CONFIG)
    views = load_views(cfg.views_path)
    view_ids = [str(v.id) for v in views]
    view_by_id = {str(v.id): v for v in views}
    program = lambda cid: ObjectProgram.from_spec(decode_spec(json.loads((bundle / f"private/{cid}.json").read_text())))
    clean = {g["asset_id"]: g["case_id"] for g in gold if g["kind"] == "clean"}
    rows = []
    with ObservationSession(cfg.render, views) as session:
        for g in sorted((g for g in gold if g["kind"] != "clean"), key=lambda g: g["case_id"]):
            record = next(j for j in screening[g["asset_id"]]["joints"] if "selected" in j)
            values, reference, candidate = record["state_values"], program(clean[g["asset_id"]]), program(g["case_id"])
            joint = record["joint_id"]
            wide = full_normal_framing(render_states(session, reference, None, joint, values, [view_ids[0]]))
            framing = replace(wide, radius=wide.radius * record["normal_padding"] / 1.5)
            ref = render_states(session, reference, framing, joint, values, view_ids)
            cand = render_states(session, candidate, framing, joint, values, view_ids)
            state = "rest" if g["defect_id"] == "structure.part_offset" else "limit"
            per_view = {}
            for v in view_ids:
                m = image_metrics(cand["frames"][state][v], ref["frames"][state][v])
                per_view[v] = {"core_changed_fraction": round(m["core_changed_fraction"], 4),
                               "largest_core_region_px": m["largest_core_region_px"],
                               "in_frame": inside_camera(cand, framing, view_by_id[v])[0],
                               "passes_gate": m["core_changed_fraction"] >= GATE_FRACTION
                               and m["largest_core_region_px"] >= GATE_REGION}
            chosen = public[g["case_id"]]["view"]
            best = max(per_view, key=lambda v: per_view[v]["core_changed_fraction"])
            rows.append({"case_id": g["case_id"], "defect_id": g["defect_id"], "asset_id": g["asset_id"],
                         "state_compared": state, "chosen_view": chosen,
                         "chosen_passes_gate": per_view[chosen]["passes_gate"],
                         "chosen_fraction": per_view[chosen]["core_changed_fraction"],
                         "best_view": best, "best_fraction": per_view[best]["core_changed_fraction"],
                         "views_passing_gate": sorted(v for v, m in per_view.items() if m["passes_gate"] and m["in_frame"]),
                         "human_one_glance": reviews[g["case_id"]]["human_judgment"]["one_glance"],
                         "human_status": reviews[g["case_id"]]["status"], "per_view": per_view})
    (bundle / "private/view_visibility.json").write_text(json.dumps({"gate": {"fraction": GATE_FRACTION, "region_px": GATE_REGION},
                                                                      "rows": rows}, indent=2), encoding="utf-8")
    for r in rows:
        print(r["case_id"], r["defect_id"].split(".")[1], "glance=", r["human_one_glance"], "chosen", r["chosen_view"],
              r["chosen_fraction"], "best", r["best_view"], r["best_fraction"], "n_views_ok", len(r["views_passing_gate"]))


if __name__ == "__main__":
    main()
