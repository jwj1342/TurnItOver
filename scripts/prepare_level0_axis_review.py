#!/usr/bin/env python3
"""Build five deliberately obvious Train-only joint-axis review candidates.

This is a visual review bundle, not a training export.  It deliberately uses a
90-degree axis change and only admits cases with large Chromium-rendered motion
differences.  The automatic screen never substitutes for the requested
one-glance human review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.prepare_obvious_defect_review import (
    DEFAULT_CONFIG, DEFAULT_SPLITS, ROOT, STATES, checker_result, image_metrics,
    render_program, sha256_path, state_values, write_jsonl,
)
from turnitover.assets.catalog import CatalogSource
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing
from turnitover.core.serde import to_dict
from turnitover.corruptions import get_corruption
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views


DEFAULT_OUT = ROOT / "docs/evidence/partnet-mobility-level0-axis-v1"
AXIS_DEG = 90.0
ATTEMPTS_PER_ASSET = 24
SELECTIONS = (
    {"asset_id": "PM_12036", "joint_id": "joint_0", "role": "refrigerator door",
     "normal_motion": "The full door swings sideways around its vertical hinge."},
    {"asset_id": "PM_12594", "joint_id": "joint_1", "role": "dishwasher door",
     "normal_motion": "The broad front door folds downward around its horizontal hinge."},
    {"asset_id": "PM_10697", "joint_id": "joint_1", "role": "laptop screen",
     "normal_motion": "The full screen tilts open and closed around the keyboard hinge."},
    {"asset_id": "PM_102016", "joint_id": "joint_0", "role": "sliding USB component",
     "normal_motion": "The large component slides straight out of and into its housing."},
    {"asset_id": "PM_5088", "joint_id": "joint_0", "role": "display panel",
     "normal_motion": "The large screen panel tilts around its stand hinge."},
)


def opaque_id(asset_id: str, kind: str, seed: int | None) -> str:
    digest = hashlib.sha256(f"level0-axis-v1:{asset_id}:{kind}:{seed}".encode()).hexdigest()
    return f"l0-{digest[:12]}"


def normal_view(render: dict) -> tuple[str, dict[str, float]]:
    scores = {
        view: min(
            image_metrics(render["frames"][state][view], render["frames"]["rest"][view])
            ["core_changed_fraction"]
            for state in ("mid", "limit")
        )
        for view in render["frames"]["rest"]
    }
    return max(scores, key=scores.get), scores


def axis_attempts(reference: ObjectProgram, joint_id: str, seed_base: int) -> list[dict]:
    corruption = get_corruption("kinematics.joint_axis", min_deg=AXIS_DEG, max_deg=AXIS_DEG)
    attempts = []
    for seed in range(seed_base, seed_base + 96):
        program, label = corruption.apply(reference, np.random.default_rng(seed))
        if label.params["joint"] == joint_id:
            attempts.append({"seed": seed, "program": program, "label": label})
        if len(attempts) == ATTEMPTS_PER_ASSET:
            return attempts
    raise RuntimeError(f"Could not obtain {ATTEMPTS_PER_ASSET} target-axis attempts for {joint_id}")


def full_normal_framing(render: dict) -> Framing:
    """Freeze a camera wide enough for every normal reference state before injection."""
    vertices = np.concatenate([
        mesh.vertices for state in STATES for mesh in render["geometry"][state].values()
    ])
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) / 2.0
    # Reserve headroom from normal geometry before injection.  This is fixed
    # solely from the clean rest/mid/limit trajectory, not adapted per defect.
    radius = float(np.linalg.norm(high - low) / 2.0) * 1.50
    return Framing(center=tuple(float(x) for x in center), radius=radius)


def border_subject_fraction(png: bytes) -> float:
    image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)
    background = image[0, 0]
    mask = np.max(np.abs(image - background), axis=2) > 16
    border = np.concatenate((mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]))
    return float(border.mean())


def screen(reference: dict, candidate: dict, view: str, check: dict) -> tuple[dict, list[str]]:
    metrics = {state: image_metrics(candidate["frames"][state][view], reference["frames"][state][view])
               for state in STATES}
    normal_motion = {state: image_metrics(reference["frames"][state][view], reference["frames"]["rest"][view])
                     for state in ("mid", "limit")}
    candidate_motion = {state: image_metrics(candidate["frames"][state][view], candidate["frames"]["rest"][view])
                        for state in ("mid", "limit")}
    reasons = []
    if metrics["rest"]["changed_fraction"] > 0.001:
        reasons.append("rest_not_matched")
    for name, rendered in (("reference", reference), ("candidate", candidate)):
        for state in STATES:
            if border_subject_fraction(rendered["frames"][state][view]) > 0.01:
                reasons.append(f"{name}_{state}_clipped")
    for state in ("mid", "limit"):
        if normal_motion[state]["core_changed_fraction"] < 0.020:
            reasons.append(f"reference_{state}_motion_too_small")
        if candidate_motion[state]["core_changed_fraction"] < 0.020:
            reasons.append(f"candidate_{state}_motion_too_small")
        if metrics[state]["core_changed_fraction"] < 0.025:
            reasons.append(f"axis_{state}_trajectory_difference_too_small")
        if metrics[state]["largest_core_region_px"] < 750:
            reasons.append(f"axis_{state}_region_too_small")
    if check["code_correct"] or not check["checks"][1]["passed"]:
        reasons.append("independent_checker_disagreement")
    return {"reference_motion": normal_motion, "candidate_motion": candidate_motion,
            "reference_candidate": metrics}, reasons


def score(metrics: dict) -> tuple[float, float]:
    states = ("mid", "limit")
    return (min(metrics["reference_candidate"][s]["core_changed_fraction"] for s in states),
            min(metrics["reference_candidate"][s]["largest_core_region_px"] for s in states))


def board(path: Path, case_id: str, reference: dict, candidate: dict, view: str) -> None:
    tile, gap, left, top = 256, 10, 104, 88
    canvas = Image.new("RGB", (left + 3 * tile + 4 * gap, top + 2 * tile + 3 * gap), "#fafaf8")
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.truetype("DejaVuSans.ttf", 16), ImageFont.truetype("DejaVuSans.ttf", 13)
    draw.text((12, 12), f"{case_id} | native 256 x 256 | pending_human_review", fill="#111", font=font)
    draw.text((12, 40), "Reference and candidate share one fixed camera. Compare the full moving part.", fill="#444", font=small)
    for col, state in enumerate(STATES):
        x = left + gap + col * (tile + gap)
        draw.text((x, 67), state, fill="#222", font=small)
        for row, (name, rendered) in enumerate((("Reference", reference), ("Candidate", candidate))):
            y = top + gap + row * (tile + gap)
            canvas.paste(Image.open(io.BytesIO(rendered["frames"][state][view])).convert("RGB"), (x, y))
            if col == 0:
                draw.text((12, y + 120), name, fill="#222", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def pair_preview(path: Path, reference: dict, candidate: dict, view: str) -> None:
    """The exact four-image rest/limit layout intended for the smallest SFT input."""
    tile, gap = 256, 10
    canvas = Image.new("RGB", (2 * tile + 3 * gap, 2 * tile + 3 * gap), "#fafaf8")
    for row, rendered in enumerate((reference, candidate)):
        for col, state in enumerate(("rest", "limit")):
            x, y = gap + col * (tile + gap), gap + row * (tile + gap)
            canvas.paste(Image.open(io.BytesIO(rendered["frames"][state][view])).convert("RGB"), (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run(config_path: Path, splits_path: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError(f"Output must not already exist: {out}")
    cfg = load_generate_config(config_path)
    if (cfg.render.width, cfg.render.height) != (256, 256):
        raise ValueError("Level 0 review must use the 2B student's 256 px input size")
    splits, catalog, views = (json.loads(splits_path.read_text()),
                              CatalogSource(cfg.asset_source["manifest"]), load_views(cfg.views_path))
    out.mkdir(parents=True)
    (out / "private").mkdir()
    public, gold, checks, screening = [], [], [], []
    with ObservationSession(cfg.render, views) as session:
        for index, selected in enumerate(SELECTIONS):
            asset_id, joint_id = selected["asset_id"], selected["joint_id"]
            if splits["groups"].get(f"PartNet-Mobility:{asset_id.removeprefix('PM_')}") != "train":
                raise ValueError(f"Selected asset is not Train: {asset_id}")
            spec, reference = catalog.get(asset_id), ObjectProgram.from_spec(catalog.get(asset_id))
            initial_ref = render_program(session, reference, None, joint_id, [str(v.id) for v in views])
            normal_framing = full_normal_framing(initial_ref)
            ref_render = render_program(session, reference, normal_framing, joint_id, [str(v.id) for v in views])
            view, normal_scores = normal_view(ref_render)
            candidates = []
            for attempt in axis_attempts(reference, joint_id, 8100 + index * 200):
                rendered = render_program(session, attempt["program"], normal_framing, joint_id, [view])
                check = checker_result(attempt["program"], rendered, reference, ref_render)
                metrics, reasons = screen(ref_render, rendered, view, check)
                candidates.append({**attempt, "render": rendered, "check": check, "metrics": metrics,
                                   "reasons": reasons})
            passing = [row for row in candidates if not row["reasons"]]
            if not passing:
                counts = Counter(reason for row in candidates for reason in row["reasons"])
                raise RuntimeError(f"No strict Level 0 axis candidate passed for {asset_id}: {dict(counts)}")
            chosen = max(passing, key=lambda row: score(row["metrics"]))
            variants = (("clean", None, reference, ref_render, None, None),
                        ("joint_axis", chosen["seed"], chosen["program"], chosen["render"],
                         chosen["label"], chosen["check"]))
            for kind, seed, program, rendered, label, check in variants:
                case_id = opaque_id(asset_id, kind, seed)
                board_path, preview_path = out / "boards" / f"{case_id}.png", out / "inputs" / f"{case_id}.png"
                board(board_path, case_id, ref_render, rendered, view)
                pair_preview(preview_path, ref_render, rendered, view)
                result = checker_result(program, rendered, reference, ref_render)
                if result["code_correct"] != (kind == "clean"):
                    raise RuntimeError(f"Independent checker disagreement for {case_id}")
                public.append({"case_id": case_id, "category": spec.category, "view": view,
                               "states": list(STATES), "input_states": ["rest", "limit"],
                               "image_size": [256, 256], "board": board_path.relative_to(out).as_posix(),
                               "input_preview": preview_path.relative_to(out).as_posix(),
                               "status": "pending_human_review"})
                gold.append({"case_id": case_id, "asset_id": asset_id, "kind": kind,
                             "expected_status": "pass" if kind == "clean" else "fail",
                             "defect_id": None if label is None else label.defect_id,
                             "joint_id": None if label is None else label.params["joint"],
                             "parts": [] if label is None else list(label.parts), "parameters": {} if label is None else label.params,
                             "candidate_sha": program.sha, "reference_sha": reference.sha,
                             "training_eligibility": "pending_human_review"})
                checks.append({"case_id": case_id, **result})
                (out / "private" / f"{case_id}.json").write_text(json.dumps(to_dict(program.spec)), encoding="utf-8")
            screening.append({"asset_id": asset_id, "joint_id": joint_id, "role": selected["role"],
                              "normal_motion": selected["normal_motion"], "view": view,
                              "normal_view_scores": normal_scores, "normal_framing": to_dict(normal_framing),
                              "axis_angle_deg": AXIS_DEG,
                              "attempts": [{"seed": row["seed"], "automatic_pass": not row["reasons"],
                                            "rejection_reasons": row["reasons"], "metrics": row["metrics"],
                                            "selected": row is chosen} for row in candidates]})
    public.sort(key=lambda row: row["case_id"])
    write_jsonl(out / "review_manifest.jsonl", public)
    write_jsonl(out / "human_review.jsonl", [{"case_id": row["case_id"], "status": row["status"],
        "human_judgment": {"verdict": None, "defect_id": None, "one_glance": None,
                           "normal_vs_candidate_motion": None, "evidence_states": [], "notes": None}}
        for row in public])
    write_jsonl(out / "private/gold.jsonl", gold)
    write_jsonl(out / "private/checks.jsonl", checks)
    (out / "private" / "screening.json").write_text(json.dumps({"assets": screening}, indent=2), encoding="utf-8")
    validation = {"status": "passed", "assets": len(SELECTIONS), "cases": len(public),
                  "variants_per_asset": 2, "pending_human_review": len(public),
                  "clean_checker_passed": sum(row["code_correct"] for row in checks),
                  "joint_axis_checker_rejected": sum(not row["code_correct"] for row in checks),
                  "axis_angle_deg": AXIS_DEG, "part_offset_cases": 0, "test_rows_touched": 0,
                  "model_calls": 0, "sft_training_runs": 0, "native_image_size": [256, 256]}
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (out / "README.md").write_text("""# Level 0 extreme joint-axis review\n\nThis is a **pending-human-review** visual bundle: five independent Train assets, each with a clean candidate and one 90-degree `kinematics.joint_axis` candidate. No `part_offset`, teacher call, SFT run, or Test row is included.\n\nOpen `boards/` for rest/mid/limit comparisons. `inputs/` is the concise four-tile input for a 2B diagnosis: reference rest/limit on the first row, candidate rest/limit on the second. All tiles are native Chromium 256x256 renders, and reference/candidate use the same frozen normal camera and state. IDs are opaque and the public manifest omits labels.\n\nFor each row in `human_review.jsonl`, decide without opening `private/`, enlarging the image, or using hierarchy: whether the candidate is correct; whether it is one-glance obvious; and one sentence describing the normal versus candidate motion direction. Keep every row pending until that review is complete. The private folder holds injection truth, specs, strict-screen diagnostics, and independent checker outputs.\n""", encoding="utf-8")
    manifest = {"version": 1, "status": "complete", "review_status": "pending_human_review",
                "scope": "Train only; Level 0 joint_axis", "inputs": {str(config_path): sha256_path(config_path),
                str(splits_path): sha256_path(splits_path), str(Path(cfg.asset_source["manifest"])): sha256_path(Path(cfg.asset_source["manifest"])),
                str(Path(__file__).resolve()): sha256_path(Path(__file__).resolve())},
                "files": {p.relative_to(out).as_posix(): sha256_path(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.splits.resolve(), args.out.resolve())))
