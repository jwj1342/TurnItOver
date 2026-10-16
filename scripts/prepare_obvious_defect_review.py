#!/usr/bin/env python3
"""Build a blind, human-review-only set of obvious PartNet-Mobility defects.

The builder reuses the registered corruptions, Chromium renderer and geometry
checker.  It searches only a few deterministic seeds per defect, renders at the
student's 256 px resolution, and emits blind comparison boards.  Automatic
gates may reject attempts but can only admit a case as pending_human_review.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from turnitover.assets.catalog import CatalogSource
from turnitover.checkers.base import Evidence
from turnitover.checkers.geometry import GeometryAlignmentChecker
from turnitover.checkers.runtime_budget import RuntimeBudgetChecker
from turnitover.config import load_generate_config
from turnitover.core.actions import Detent, RuntimeProperty, detent_value
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.corruptions import get_corruption
from turnitover.oracle.evidence import ExposureThresholds
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/generate_partnet_mobility_supervision.yaml"
DEFAULT_SPLITS = ROOT / "data/assets/partnet-mobility-trainval-v1/splits.json"
DEFAULT_OUT = ROOT / "docs/evidence/partnet-mobility-obvious-defects-v1"
SELECTIONS = (
    {"asset_id": "PM_12036", "joint_id": "joint_0", "role": "cabinet door"},
    {"asset_id": "PM_25308", "joint_id": "joint_1", "role": "drawer"},
    {"asset_id": "PM_102257", "joint_id": "joint_0", "role": "flip lid"},
)
STATES = ("rest", "mid", "limit")
TARGETED_ATTEMPTS = 3
MAX_SEEDS = 48
AXIS_DEG = 60.0
OFFSET_M = 0.18
SEMANTIC_NOTES = {
    "PM_12036": {
        "reference": "The full-height door swings laterally through a large arc.",
        "joint_axis": "At mid and limit the full door leaves the lateral hinge plane and flips upward/outward.",
        "part_offset": "The full door is displaced relative to the cabinet at rest and remains displaced while moving.",
    },
    "PM_25308": {
        "reference": "The large drawer translates horizontally out of the furniture body.",
        "joint_axis": "The drawer moves upward/inward instead of following the horizontal slide direction.",
        "part_offset": "The complete drawer box is vertically and laterally misregistered in all three states.",
    },
    "PM_102257": {
        "reference": "The large front lid follows a clear revolute opening arc.",
        "joint_axis": "The lid takes a strongly different out-of-plane arc at mid and limit.",
        "part_offset": "The complete lid is shifted relative to the can; the mismatch is clearest at mid and limit.",
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def blind_id(asset_id: str, kind: str, seed: int | None) -> str:
    raw = f"obvious-defect-v1:{asset_id}:{kind}:{seed}".encode()
    return f"odr-{hashlib.sha256(raw).hexdigest()[:12]}"


def image_metrics(candidate: bytes, reference: bytes) -> dict:
    cand = np.asarray(Image.open(io.BytesIO(candidate)).convert("RGB"))
    ref = np.asarray(Image.open(io.BytesIO(reference)).convert("RGB"))
    delta = np.abs(cand.astype(np.float32) - ref.astype(np.float32)).max(axis=2) / 255.0
    mask = delta > ExposureThresholds().pixel_delta
    core = ndimage.binary_erosion(mask, structure=np.ones((5, 5)), border_value=0)
    labels, count = ndimage.label(core)
    sizes = np.bincount(labels.ravel())[1:] if count else np.array([0])
    return {
        "changed_fraction": float(mask.mean()),
        "core_changed_fraction": float(core.mean()),
        "largest_core_region_px": int(sizes.max()),
    }


def candidate_attempts(reference: ObjectProgram, target_joint: str, kind: str,
                       seed_base: int) -> tuple[list[dict], list[dict]]:
    """Use the production corruption and retain a few attempts hitting the target part/joint."""
    target_part = reference.spec.joint(target_joint).part
    params = ({"min_deg": AXIS_DEG, "max_deg": AXIS_DEG} if kind == "joint_axis"
              else {"min_m": OFFSET_M, "max_m": OFFSET_M})
    corruption_id = "kinematics.joint_axis" if kind == "joint_axis" else "structure.part_offset"
    corruption = get_corruption(corruption_id, **params)
    accepted, rejected = [], []
    for offset in range(MAX_SEEDS):
        seed = seed_base + offset
        program, label = corruption.apply(reference, np.random.default_rng(seed))
        hit = (label.params.get("joint") == target_joint if kind == "joint_axis"
               else label.parts == (target_part,))
        if not hit:
            rejected.append({"kind": kind, "seed": seed, "reason": "wrong_target",
                             "actual_joint": label.params.get("joint"), "actual_parts": list(label.parts)})
            continue
        accepted.append({"kind": kind, "seed": seed, "program": program, "label": label})
        if len(accepted) == TARGETED_ATTEMPTS:
            break
    if len(accepted) != TARGETED_ATTEMPTS:
        raise RuntimeError(f"Could not find {TARGETED_ATTEMPTS} targeted {kind} attempts")
    return accepted, rejected


def state_values(joint: dict) -> dict[str, float]:
    limits = tuple(joint["limits"])
    return {"rest": 0.0, "mid": detent_value(Detent.MID, limits),
            "limit": detent_value(Detent.LIMIT, limits)}


def render_program(session: ObservationSession, program: ObjectProgram, framing,
                   joint_id: str, view_ids: list[str]) -> dict:
    info = session.load(program, framing=framing)
    values = state_values(info.joints[joint_id])
    frames, geometry = {}, {}
    try:
        for state in STATES:
            session.reset_joints()
            if state != "rest":
                session.actuate_joint(joint_id, values[state])
            geometry[state] = session.export_geometry()
            frames[state] = {view: session.request_view(view).png for view in view_ids}
        stats = session.query_runtime(RuntimeProperty.STATS)
        hierarchy = session.query_runtime(RuntimeProperty.HIERARCHY)
        return {"frames": frames, "geometry": geometry, "stats": stats,
                "hierarchy": hierarchy, "framing": info.framing, "joint": info.joints[joint_id],
                "parts": info.parts}
    finally:
        session.dispose()


def measure_attempts(reference_render: dict, attempts: list[dict], view_ids: list[str]) -> None:
    for attempt in attempts:
        per_view = {}
        for view in view_ids:
            by_state = {state: image_metrics(attempt["render"]["frames"][state][view],
                                             reference_render["frames"][state][view])
                        for state in STATES}
            per_view[view] = by_state
        attempt["metrics"] = per_view


def choose_triplet(reference_render: dict, axes: list[dict], offsets: list[dict],
                   view_ids: list[str]) -> tuple[str, dict, dict, dict]:
    rows = []
    best = None
    for axis in axes:
        for offset in offsets:
            for view in view_ids:
                motion = max(image_metrics(reference_render["frames"][state][view],
                                           reference_render["frames"]["rest"][view])
                             ["core_changed_fraction"] for state in ("mid", "limit"))
                axis_core = max(axis["metrics"][view][state]["core_changed_fraction"]
                                for state in ("mid", "limit"))
                offset_core = min(offset["metrics"][view][state]["core_changed_fraction"]
                                  for state in STATES)
                axis_region = max(axis["metrics"][view][state]["largest_core_region_px"]
                                  for state in ("mid", "limit"))
                offset_region = min(offset["metrics"][view][state]["largest_core_region_px"]
                                    for state in STATES)
                passed = (motion >= 0.01 and axis_core >= 0.02 and offset_core >= 0.01
                          and axis_region >= 500 and offset_region >= 300)
                score = min(motion / 0.01, axis_core / 0.02, offset_core / 0.01) + \
                    0.0001 * min(axis_region, offset_region)
                row = {"view": view, "axis_seed": axis["seed"], "offset_seed": offset["seed"],
                       "reference_motion_core": motion, "axis_core": axis_core,
                       "offset_core": offset_core, "axis_region_px": axis_region,
                       "offset_region_px": offset_region, "passed": passed, "score": score}
                rows.append(row)
                if passed and (best is None or score > best[0]):
                    best = (score, view, axis, offset, row)
    if best is None:
        raise RuntimeError("No axis/offset/view triplet passed the 256 px visibility gate")
    _, view, axis, offset, selected = best
    selected_views = []
    for row in rows:
        if row["axis_seed"] == axis["seed"] and row["offset_seed"] == offset["seed"]:
            reason = ("selected" if row["view"] == view else
                      "eligible_not_selected" if row["passed"] else "visibility_gate_failed")
            selected_views.append({**row, "reason": reason})
    attempts = []
    for kind, candidates, chosen in (("joint_axis", axes, axis), ("part_offset", offsets, offset)):
        for candidate in candidates:
            states = ("mid", "limit") if kind == "joint_axis" else STATES
            seed_key = "axis_seed" if kind == "joint_axis" else "offset_seed"
            eligible_scores = [row["score"] for row in rows
                               if row[seed_key] == candidate["seed"] and row["passed"]]
            reason = ("selected" if candidate is chosen else
                      "not_selected_after_stable_tiebreak"
                      if eligible_scores and math.isclose(max(eligible_scores), best[0])
                      else "lower_combined_visibility_score")
            attempts.append({"kind": kind, "seed": candidate["seed"],
                             "status": "selected" if candidate is chosen else "rejected",
                             "reason": reason,
                             "best_core_fraction": max(candidate["metrics"][v][s]["core_changed_fraction"]
                                                       for v in view_ids for s in states),
                             "best_core_region_px": max(candidate["metrics"][v][s]["largest_core_region_px"]
                                                        for v in view_ids for s in states)})
    audit = {"selected": {**selected, "status": "pending_human_review"},
             "combination_counts": {"total": len(rows), "passed": sum(row["passed"] for row in rows),
                                    "visibility_gate_failed": sum(not row["passed"] for row in rows),
                                    "eligible_not_selected": sum(row["passed"] for row in rows) - 1},
             "targeted_attempts": attempts, "selected_pair_views": selected_views}
    return view, axis, offset, audit


def checker_result(candidate: ObjectProgram, rendered: dict, reference: ObjectProgram,
                   reference_render: dict) -> dict:
    def evidence(program, value):
        return Evidence(program.sha, True, None,
                        {"rest" if state == "rest" else f"joint:target:{state}": geo
                         for state, geo in value["geometry"].items()},
                        value["stats"], value["hierarchy"])
    observed = evidence(candidate, rendered)
    gold = evidence(reference, reference_render)
    geometry = to_dict(GeometryAlignmentChecker(samples_per_part=1000).check(observed, gold))
    runtime = to_dict(RuntimeBudgetChecker(1_000_000, 1024).check(observed, None))
    return {"candidate_sha": candidate.sha, "reference_sha": reference.sha,
            "privileged": True, "used_by_student": False,
            "checks": [geometry, runtime], "code_correct": geometry["passed"] and runtime["passed"]}


def make_board(path: Path, case_id: str, spec, joint, view: str,
               reference_frames: dict, candidate_frames: dict) -> None:
    tile, gap, left, top = 256, 10, 112, 106
    canvas = Image.new("RGB", (left + 3 * tile + 4 * gap, top + 2 * tile + 3 * gap), "#fafaf8")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
    small = ImageFont.truetype("DejaVuSans.ttf", 14)
    values = state_values(joint)
    title = f"{case_id}  |  {spec.asset_id} {spec.category}  |  view {view}"
    draw.text((12, 12), title, fill="#111111", font=font)
    draw.text((12, 40), (f"Joint: {joint['id']} ({joint['type']})   Moving part: {joint['part']}   "
                         "Status: pending_human_review"), fill="#333333", font=small)
    draw.text((12, 64), "Compare motion semantics at native 256 x 256; no defect answer is shown.",
              fill="#555555", font=small)
    for col, state in enumerate(STATES):
        x = left + gap + col * (tile + gap)
        draw.text((x, 82), f"{state} ({values[state]:.3f})", fill="#222222", font=small)
        for row, (label, frames) in enumerate((("Reference", reference_frames),
                                               ("Candidate", candidate_frames))):
            y = top + gap + row * (tile + gap)
            canvas.paste(Image.open(io.BytesIO(frames[state][view])).convert("RGB"), (x, y))
            if col == 0:
                draw.text((12, y + tile // 2 - 8), label, fill="#222222", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def run(config_path: Path, splits_path: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError(f"Output must not exist: {out}")
    cfg = load_generate_config(config_path)
    if (cfg.render.width, cfg.render.height) != (256, 256):
        raise ValueError("This review set must match the student's 256 x 256 resolution")
    splits = json.loads(splits_path.read_text())
    catalog = CatalogSource(cfg.asset_source["manifest"])
    views = load_views(cfg.views_path)
    view_ids = [str(view.id) for view in views]
    out.mkdir(parents=True)
    (out / "private").mkdir()
    public, gold, checks, screening = [], [], [], []

    with ObservationSession(cfg.render, views) as session:
        for asset_index, selected in enumerate(SELECTIONS):
            asset_id, joint_id = selected["asset_id"], selected["joint_id"]
            source_asset_id = asset_id.removeprefix("PM_")
            if splits["groups"].get(f"PartNet-Mobility:{source_asset_id}") != "train":
                raise ValueError(f"Selected asset is not Train: {asset_id}")
            spec = catalog.get(asset_id)
            reference = ObjectProgram.from_spec(spec)
            reference_render = render_program(session, reference, None, joint_id, view_ids)
            axes, axis_rejected = candidate_attempts(reference, joint_id, "joint_axis", 1100 + asset_index * 100)
            offsets, offset_rejected = candidate_attempts(reference, joint_id, "part_offset", 2100 + asset_index * 100)
            for attempt in axes + offsets:
                attempt["render"] = render_program(session, attempt["program"], reference_render["framing"],
                                                    joint_id, view_ids)
            measure_attempts(reference_render, axes, view_ids)
            measure_attempts(reference_render, offsets, view_ids)
            view, axis, offset, visibility = choose_triplet(reference_render, axes, offsets, view_ids)
            screening.append({"asset_id": asset_id, "role": selected["role"],
                              "semantic_prescreen": {"reviewer": "AI operator", "final_human_review": "required",
                                                     **SEMANTIC_NOTES[asset_id]},
                              "wrong_target_attempts": axis_rejected + offset_rejected,
                              "visibility_screening": visibility})
            variants = [("clean", None, reference, reference_render),
                        ("joint_axis", axis["seed"], axis["program"], axis["render"]),
                        ("part_offset", offset["seed"], offset["program"], offset["render"])]
            for kind, seed, candidate, candidate_render in variants:
                case_id = blind_id(asset_id, kind, seed)
                board = out / "boards" / f"{case_id}.png"
                joint = reference_render["joint"]
                make_board(board, case_id, spec, joint, view,
                           reference_render["frames"], candidate_render["frames"])
                result = checker_result(candidate, candidate_render, reference, reference_render)
                expected_correct = kind == "clean"
                if result["code_correct"] != expected_correct:
                    raise RuntimeError(f"Independent checker disagrees for {case_id}")
                public.append({"case_id": case_id, "asset_id": asset_id, "category": spec.category,
                               "joint_id": joint_id, "joint_type": joint["type"],
                               "moving_part": joint["part"], "semantic_role": selected["role"],
                               "view": view, "states": list(STATES), "image_size": [256, 256],
                               "board": board.relative_to(out).as_posix(), "board_sha256": sha256_path(board),
                               "status": "pending_human_review", "independent_checker": "complete"})
                label = None if kind == "clean" else (axis["label"] if kind == "joint_axis" else offset["label"])
                gold.append({"case_id": case_id, "kind": kind, "expected_status": "pass" if kind == "clean" else "fail",
                             "defect_id": None if label is None else label.defect_id,
                             "joint_id": None if label is None else label.params.get("joint"),
                             "parts": [] if label is None else list(label.parts),
                             "parameters": {} if label is None else dict(label.params),
                             "label": {} if label is None else dict(label.label), "seed": seed,
                             "candidate_sha": candidate.sha, "reference_sha": reference.sha,
                             "training_eligibility": "pending_human_review"})
                checks.append({"case_id": case_id, **result})

    write_jsonl(out / "review_manifest.jsonl", public)
    write_jsonl(out / "human_review.jsonl", [
        {"case_id": row["case_id"], "status": "pending_human_review",
         "human_judgment": {"verdict": None, "defect_id": None, "joint_id": None,
                            "parts": [], "evidence_states": [], "notes": None}}
        for row in public])
    write_jsonl(out / "private/gold.jsonl", gold)
    write_jsonl(out / "private/checks.jsonl", checks)
    write_json_atomic(out / "private/screening.json", {"assets": screening})
    write_json_atomic(out / "validation.json", {
        "status": "passed", "cases": len(public), "train_assets": len(SELECTIONS),
        "pending_human_review": sum(row["status"] == "pending_human_review" for row in public),
        "clean_checker_passed": sum(row["code_correct"] for row in checks),
        "single_defect_checker_rejected": sum(not row["code_correct"] for row in checks),
        "native_image_size": [256, 256], "public_answer_fields": [],
        "test_rows_touched": 0, "model_calls": 0, "sft_training_runs": 0,
    })
    (out / "REPRODUCE.md").write_text(
        "# PartNet-Mobility obvious-defect review bundle\n\n"
        "This bundle contains nine blind comparison boards: three Train assets, each with three "
        "candidate variants. Every board remains `pending_human_review`; automated gates do not "
        "promote a candidate to training data. Gold labels, generation parameters, independent "
        "checker details, and rejected attempts are under `private/`. No Kimi/model call or SFT "
        "training is part of this workflow, and Test is not read or modified.\n\n"
        "Each board shows reference and candidate at rest, mid, and limit using native 256 x 256 "
        "Chromium tiles. It names the asset, selected joint, joint type, moving part, and view, but "
        "does not show the defect answer. Fill `human_review.jsonl` only after deciding whether the "
        "error type, joint, part, and evidence frames are unambiguous.\n\n"
        "## Reproduce\n\n"
        "Run from the repository root into a fresh output directory:\n\n"
        "```bash\n"
        "repro_root=$(mktemp -d /tmp/turnitover-obvious-defects-repro.XXXXXX)\n"
        "python -m scripts.prepare_obvious_defect_review --out \"$repro_root/bundle\"\n"
        "diff -rq docs/evidence/partnet-mobility-obvious-defects-v1 \"$repro_root/bundle\"\n"
        "```\n\n"
        "## Validate\n\n"
        "```bash\n"
        "python -m pytest tests/unit/test_obvious_defect_review.py -q\n"
        "python -m pytest tests/unit -q\n"
        "python -m py_compile scripts/prepare_obvious_defect_review.py\n"
        "git diff --check\n"
        "```\n",
        encoding="utf-8")
    manifest = {
        "version": 1, "status": "complete", "review_status": "pending_human_review",
        "scope": "PartNet-Mobility Train only", "cases": len(public), "assets": len(SELECTIONS),
        "variants_per_asset": 3, "student_image_size": [256, 256], "model_calls": 0,
        "sft_training_runs": 0, "test_rows_touched": 0,
        "catalog": str(Path(cfg.asset_source["manifest"]).resolve()),
        "catalog_sha256": sha256_path(Path(cfg.asset_source["manifest"])),
        "splits": str(splits_path.resolve()), "splits_sha256": sha256_path(splits_path),
        "config": str(config_path.resolve()), "config_sha256": sha256_path(config_path),
        "generation": {"axis_angle_deg": AXIS_DEG, "part_offset_m": OFFSET_M,
                       "targeted_attempts_per_defect": TARGETED_ATTEMPTS,
                       "corruptions": ["kinematics.joint_axis", "structure.part_offset"]},
        "files": {path.relative_to(out).as_posix(): sha256_path(path)
                  for path in sorted(out.rglob("*")) if path.is_file()},
    }
    write_json_atomic(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.splits.resolve(), args.out.resolve())
    print(json.dumps({"status": result["status"], "cases": result["cases"],
                      "review_status": result["review_status"], "out": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
