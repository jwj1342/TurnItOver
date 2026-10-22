#!/usr/bin/env python3
"""Pilot construction for issue #9: direction_flip (Easy/Medium) and part_offset (Medium) review samples.

Each selected object yields one clean control and one defective sample sharing a frozen normal-only camera,
joint state and views. Automatic gates never substitute for the human review: every row is written as
``pending_human_review`` and the review page is blind (opaque ids, no labels). No model calls are made.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.level0_expansion_search import attempts as axis_attempt_list, inside_camera, render_states
from scripts.level0_expansion_sources import load_pool
from scripts.prepare_level0_axis_review import board, border_subject_fraction, full_normal_framing, normal_view, pair_preview, screen
from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, ROOT, STATES, checker_result, image_metrics, sha256_path, write_jsonl
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.urdf import UrdfModel
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.corruptions import get_corruption
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

OUT = ROOT / "docs/evidence/construction-pilot-v1"
ROUND3 = ROOT / "data/assets/partnet-mobility-level0-round3/catalog.json"
ROUND3_DOWNLOAD = ROOT / "data/partnet-mobility-level0-round3/download-manifest.json"
SALT = "construction-pilot-v1"
PER_CONFIG = 5
MAX_OBJECTS_PER_CONFIG = 60
MAX_PER_CATEGORY = 2
OFFSET_FRACTION = 0.12  # offset magnitude as a fraction of the reference bounding-box diagonal
CONFIGS = (
    {"name": "easy_direction_flip", "difficulty": "easy", "defect": "kinematics.direction_flip", "joints": "single"},
    {"name": "medium_direction_flip", "difficulty": "medium", "defect": "kinematics.direction_flip", "joints": "multi"},
    {"name": "medium_part_offset", "difficulty": "medium", "defect": "structure.part_offset", "joints": "single"},
)


def _hash(*parts) -> str:
    return hashlib.sha256(":".join(str(p) for p in (SALT, *parts)).encode()).hexdigest()


def opaque_id(asset_id: str, kind: str, seed) -> str:
    return f"cp-{_hash(asset_id, kind, seed)[:12]}"


def flip_attempts(reference, joint, ref, seed_base):
    corruption = get_corruption("kinematics.direction_flip")
    for seed in range(seed_base, seed_base + 4096):
        program, label = corruption.apply(reference, np.random.default_rng(seed))
        if label.params["joint"] == joint.id:
            return [{"program": program, "label": label, "seed": seed}]
    return []


def offset_attempts(reference, joint, ref, seed_base):
    points = np.concatenate([m.vertices for m in ref["geometry"]["rest"].values()])
    magnitude = OFFSET_FRACTION * float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    corruption = get_corruption("structure.part_offset", min_m=magnitude, max_m=magnitude)
    found = []
    for seed in range(seed_base, seed_base + 8192):
        program, label = corruption.apply(reference, np.random.default_rng(seed))
        if label.parts == (joint.part,):
            found.append({"program": program, "label": label, "seed": seed})
        if len(found) == 24:
            break
    return found


def _wrong_part(check, label) -> bool:
    failed = {p for c in check["checks"][:1] for inv in c["invariants"] if not inv["passed"] for p in inv["parts"]}
    return not set(label.parts) <= failed


def axis_attempts(reference, joint, ref, seed_base):
    return axis_attempt_list(reference, joint.id, seed_base)


def screen_motion(ref, rendered, view, check, label):
    metrics, reasons = screen(ref, rendered, view, check)
    if _wrong_part(check, label):
        reasons.append("checker_does_not_flag_target_part")
    return metrics, reasons


def screen_offset(ref, rendered, view, check, label):
    rest = image_metrics(rendered["frames"]["rest"][view], ref["frames"]["rest"][view])
    motion = {s: image_metrics(rendered["frames"][s][view], rendered["frames"]["rest"][view]) for s in ("mid", "limit")}
    reasons = []
    if rest["core_changed_fraction"] < 0.025:
        reasons.append("offset_rest_difference_too_small")
    if rest["largest_core_region_px"] < 750:
        reasons.append("offset_rest_region_too_small")
    for state in STATES:
        if border_subject_fraction(rendered["frames"][state][view]) > 0.01:
            reasons.append(f"candidate_{state}_clipped")
    for state in ("mid", "limit"):
        if motion[state]["core_changed_fraction"] < 0.020:
            reasons.append(f"candidate_{state}_motion_too_small")
    if check["code_correct"] or not check["checks"][1]["passed"]:
        reasons.append("independent_checker_disagreement")
    elif _wrong_part(check, label):
        reasons.append("checker_does_not_flag_target_part")
    return {"reference_candidate_rest": rest, "candidate_motion": motion}, reasons


def find_case(session, reference, views, config, seed_base):
    """First joint whose frozen normal configuration admits a passing attempt; no per-defect camera retry."""
    view_ids = [str(v.id) for v in views]
    view_by_id = {str(v.id): v for v in views}
    make_attempts, screen_fn = HANDLERS[config["defect"]]
    diagnostics = []
    for index, joint in enumerate(reference.spec.joints):
        endpoints = sorted(set(joint.limits) - {0.}, key=lambda x: (-abs(x), -x))
        normal = None
        for endpoint in endpoints:
            values = {"rest": 0., "mid": endpoint / 2., "limit": endpoint}
            wide = full_normal_framing(render_states(session, reference, None, joint.id, values, [view_ids[0]]))
            for padding in (1.25, 1.0, 1.5):
                framing = replace(wide, radius=wide.radius * padding / 1.5)
                ref = render_states(session, reference, framing, joint.id, values, view_ids)
                _, scores = normal_view(ref)
                eligible = [v for v in sorted(scores, key=lambda v: (-scores[v], v))
                            if scores[v] >= .020 and inside_camera(ref, framing, view_by_id[v])[0]]
                record = {"joint_id": joint.id, "state_values": values, "normal_padding": padding,
                          "normal_view_scores": scores, "attempts": []}
                diagnostics.append(record)
                if not eligible:
                    record["rejection_reasons"] = ["normal_motion_or_framing_gate_failed"]
                    continue
                if not checker_result(reference, ref, reference, ref)["code_correct"]:
                    record["rejection_reasons"] = ["clean_checker_failed"]
                    continue
                normal = ref, framing, values, eligible[0], record
                break
            if normal:
                break
        if not normal:
            continue
        ref, framing, values, view, record = normal
        attempts = make_attempts(reference, joint, ref, seed_base + index * 10000)
        if not attempts:
            record["rejection_reasons"] = ["no_attempt_targets_joint_or_part"]
        for attempt in attempts:
            rendered = render_states(session, attempt["program"], framing, joint.id, values, [view])
            check = checker_result(attempt["program"], rendered, reference, ref)
            metrics, reasons = screen_fn(ref, rendered, view, check, attempt["label"])
            in_frame, extent = inside_camera(rendered, framing, view_by_id[view])
            if not in_frame:
                reasons.append("geometry_outside_frozen_camera")
            record["attempts"].append({"seed": attempt["seed"], "view": view, "automatic_pass": not reasons,
                                       "rejection_reasons": reasons, "projected_extent_ndc": extent, "metrics": metrics})
            if not reasons:
                record["selected"] = {"seed": attempt["seed"], "view": view}
                return (ref, {**attempt, "render": rendered, "view": view, "framing": framing}, joint.id, values), diagnostics
    return None, diagnostics


HANDLERS = {
    "kinematics.direction_flip": (flip_attempts, screen_motion),
    "kinematics.joint_axis": (axis_attempts, screen_motion),
    "structure.part_offset": (offset_attempts, screen_offset),
}


def candidate_order(pool, config, taken):
    want_single = config["joints"] == "single"
    rows = [r for r in pool if (len(r["spec"].joints) == 1) == want_single and r["spec"].asset_id not in taken]
    return sorted(rows, key=lambda r: _hash(config["name"], r["spec"].asset_id))


def write_frames(out, case_id, ref, rendered, view):
    for index, (data, state) in enumerate(((ref, "rest"), (ref, "limit"), (rendered, "rest"), (rendered, "limit"))):
        path = out / "frames" / case_id / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data["frames"][state][view])


def review_page(out, rows):
    cards = "\n".join(
        f'<section><h3>{r["case_id"]}</h3><img src="{r["board"]}" width="100%"></section>' for r in rows)
    (out / "review.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Construction pilot review</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto}}section{{margin:2em 0;border-bottom:1px solid #ccc}}</style>
<h1>Construction pilot：盲审</h1>
<p>每张图板：上排参考、下排候选；列依次为 rest / mid / limit，同一固定相机。请先不要打开 private/。
对每个 case_id，在 human_review.jsonl 中填写：候选是否正确（verdict: pass/fail）、一眼是否明显、哪里不对（一句话，运动方向或位置）、
有无出框或遮挡、被指出的部件是否就是你认为出错的部件。</p>
{cards}""", encoding="utf-8")


def run(out: Path, per_config: int) -> dict:
    if out.exists():
        raise ValueError(f"Output must not already exist: {out}")
    cfg = load_generate_config(DEFAULT_CONFIG)
    if (cfg.render.width, cfg.render.height) != (256, 256):
        raise ValueError("pilot requires native 256x256 rendering")
    pool, _, provenance = load_pool(catalog_path=ROUND3, download_path=ROUND3_DOWNLOAD)
    views = load_views(cfg.views_path)
    for sub in ("private", "boards", "inputs", "frames"):
        (out / sub).mkdir(parents=True)
    public, gold, checks, screening, taken = [], [], [], [], set()
    summary = {}
    with ObservationSession(cfg.render, views) as session:
        for config in CONFIGS:
            accepted, categories, tried = [], Counter(), 0
            for index, item in enumerate(candidate_order(pool, config, taken)):
                spec, meta = item["spec"], item["metadata"]
                if len(accepted) >= per_config or tried >= MAX_OBJECTS_PER_CONFIG:
                    break
                if categories[spec.category] >= MAX_PER_CATEGORY:
                    continue
                tried += 1
                reference = ObjectProgram.from_spec(spec)
                pair, diagnostics = find_case(session, reference, views, config, 2_000_000 + index * 100_000)
                authenticity = None
                if pair:
                    source = Path(meta["source_urdf"])
                    try:
                        authenticity = audit_model(session, UrdfModel(source), spec)
                    except Exception as error:  # keep the raw failure, never infer fidelity
                        authenticity = {"passed": False, "error_type": type(error).__name__, "error": str(error)}
                    if authenticity.get("passed") is not True:
                        pair = None
                screening.append({"config": config["name"], "asset_id": spec.asset_id, "category": spec.category,
                                  "status": "automatic_pass" if pair else "rejected",
                                  "source_authenticity": authenticity, "joints": diagnostics})
                print(json.dumps({"config": config["name"], "asset": spec.asset_id, "pass": pair is not None,
                                  "accepted": len(accepted) + bool(pair)}), flush=True)
                if not pair:
                    continue
                ref, chosen, joint_id, values = pair
                accepted.append(spec.asset_id), taken.add(spec.asset_id)
                categories[spec.category] += 1
                for kind, seed, program, rendered, label in (
                        ("clean", None, reference, ref, None),
                        (config["defect"].split(".")[1], chosen["seed"], chosen["program"], chosen["render"], chosen["label"])):
                    case_id = opaque_id(spec.asset_id, kind, seed)
                    check = checker_result(program, rendered, reference, ref)
                    if check["code_correct"] != (label is None):
                        raise ValueError(f"checker disagreement: {case_id}")
                    board(out / "boards" / f"{case_id}.png", case_id, ref, rendered, chosen["view"])
                    pair_preview(out / "inputs" / f"{case_id}.png", ref, rendered, chosen["view"])
                    write_frames(out, case_id, ref, rendered, chosen["view"])
                    public.append({"case_id": case_id, "category": spec.category, "view": chosen["view"],
                                   "states": list(STATES), "input_states": ["rest", "limit"], "image_size": [256, 256],
                                   "board": f"boards/{case_id}.png", "input_preview": f"inputs/{case_id}.png",
                                   "images": [f"frames/{case_id}/{i}.png" for i in range(4)],
                                   "prompt": "Images: reference rest, reference limit, candidate rest, candidate limit. "
                                             "Is the candidate correct? Return pass or fail and describe the problem.",
                                   "status": "pending_human_review"})
                    gold.append({"case_id": case_id, "asset_id": spec.asset_id, "config": config["name"],
                                 "difficulty": config["difficulty"], "kind": "clean" if label is None else kind,
                                 "expected_status": "pass" if label is None else "fail",
                                 "defect_id": None if label is None else label.defect_id,
                                 "joint_id": joint_id if label is not None and label.defect_id.startswith("kinematics.") else None,
                                 "parts": [] if label is None else list(label.parts),
                                 "parameters": {} if label is None else label.params,
                                 "state_values": values, "candidate_sha": program.sha, "reference_sha": reference.sha,
                                 "training_eligibility": "pending_human_review"})
                    checks.append({"case_id": case_id, **check})
                    (out / "private" / f"{case_id}.json").write_text(json.dumps(to_dict(program.spec)), encoding="utf-8")
            summary[config["name"]] = {"objects_tried": tried, "objects_accepted": len(accepted), "target": per_config}
    public.sort(key=lambda r: r["case_id"])
    write_jsonl(out / "review_manifest.jsonl", public)
    write_jsonl(out / "human_review.jsonl", [{"case_id": r["case_id"], "status": "pending_human_review", "reviewer": None,
        "reviewed_at": None, "human_judgment": {"verdict": None, "one_glance": None, "what_is_wrong": None,
        "no_clipping_or_occlusion": None, "flagged_part_is_the_wrong_one": None, "notes": None}} for r in public])
    write_jsonl(out / "private/gold.jsonl", sorted(gold, key=lambda g: g["case_id"]))
    write_jsonl(out / "private/checks.jsonl", checks)
    (out / "private/screening.json").write_text(json.dumps({"assets": screening}, indent=2), encoding="utf-8")
    review_page(out, public)
    validation = {"status": "automatic_checks_passed", "review_status": "pending_human_review", "configs": summary,
                  "cases": len(public), "clean": sum(g["kind"] == "clean" for g in gold),
                  "by_defect": dict(Counter(g["defect_id"] for g in gold if g["defect_id"])),
                  "model_calls": 0, "native_image_size": [256, 256]}
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    manifest = {"version": 1, "inputs": {**provenance, str(Path(__file__).resolve()): sha256_path(Path(__file__).resolve())},
                "files": {p.relative_to(out).as_posix(): sha256_path(p) for p in sorted(out.rglob("*"))
                          if p.is_file() and p.name != "manifest.json"}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--per-config", type=int, default=PER_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.out.resolve(), args.per_config)))
