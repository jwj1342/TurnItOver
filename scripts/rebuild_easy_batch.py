#!/usr/bin/env python3
"""Fix the code-level shortcuts of easy-batch-v1 (issue #9, issue #8 item 9). No model calls.

``flip``: the 29 ``direction_flip`` candidates are regenerated with the axis negated instead of ``sign``. The motion
is identical, so each candidate is re-rendered under the frozen camera and its four images must hash exactly like
the committed ones; only then is the new spec written and the gold row updated. Reviews of the images stay valid.
``axis``: ``joint_axis`` samples are rebuilt with the ``axis_permutation`` mode and the unchanged pixel gates and
written as a new batch (easy-batch-v2); clean controls stay the easy-batch-v1 ones and their reference images must
match. Objects whose reference axis is not a coordinate axis are skipped with the reason recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts import construct_easy_batch as v1
from scripts.audit_pilot_view_visibility import per_view_visibility
from scripts.construction_pilot import find_case, write_frames
from scripts.level0_expansion_search import render_states
from scripts.prepare_level0_axis_review import board, pair_preview
from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, ROOT, STATES, checker_result, sha256_path
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import from_dict, to_dict
from turnitover.core.spec import AssetSpec
from turnitover.corruptions import get_corruption
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

OUT_V2 = ROOT / "docs/evidence/easy-batch-v2"
SPECS_V2 = ROOT / "data/benchmarks/easy-batch-v2-specs"
SALT_V2 = "easy-batch-v2"
SEED_BASE_V2 = 4_000_000
HANDLER = "kinematics.joint_axis/axis_permutation"
SPOT_CHECK = 10


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_spec(path: Path) -> AssetSpec:
    return from_dict(AssetSpec, json.loads(path.read_text(encoding="utf-8")))


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def rendered_hashes(ref: dict, rendered: dict, view: str) -> list[str]:
    return [sha256_bytes(data["frames"][state][view]) for data, state in
            ((ref, "rest"), (ref, "limit"), (rendered, "rest"), (rendered, "limit"))]


def committed_hashes(out: Path, case_id: str) -> list[str]:
    return [sha256_path(out / "frames" / case_id / f"{i}.png") for i in range(4)]


def refresh_manifest(out: Path, specs: Path, scripts: tuple[Path, ...]) -> None:
    """Re-pin every generated file; ``human_review.jsonl`` stays unpinned because the reviewer edits it."""
    manifest = {"version": 1, "inputs": {str(p.resolve()): sha256_path(p) for p in scripts},
                "files": {p.relative_to(out).as_posix(): sha256_path(p) for p in sorted(out.rglob("*"))
                          if p.is_file() and p.name not in ("manifest.json", "human_review.jsonl")},
                "specs_outside_git": str(specs),
                "spec_sha256": {p.name: sha256_path(p) for p in sorted(specs.glob("*.json"))},
                "note": "human_review.jsonl is the only file edited after generation and is not hash-pinned."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def session_context():
    cfg = load_generate_config(DEFAULT_CONFIG)
    if (cfg.render.width, cfg.render.height) != (256, 256):
        raise ValueError("requires native 256x256 rendering")
    views = load_views(cfg.views_path)
    return cfg, views


def regenerate_direction_flip(out: Path, specs: Path) -> dict:
    gold = v1.rows(out / "private/gold.jsonl")
    cfg, views = session_context()
    records, updated = [], {}
    with ObservationSession(cfg.render, views) as session:
        for row in (g for g in gold if g["kind"] == "direction_flip"):
            cid = row["case_id"]
            reference = ObjectProgram.from_spec(load_spec(specs / f"{row['asset_id']}.reference.json"))
            old_spec = load_spec(specs / f"{cid}.json")
            record = {"case_id": cid, "asset_id": row["asset_id"], "old_candidate_sha": row["candidate_sha"],
                      "old_sign": old_spec.joints[0].sign, "old_axis": list(old_spec.joints[0].axis)}
            pair, _ = find_case(session, reference, views, {"defect": "kinematics.direction_flip"}, row["seed"])
            if not pair or pair[1]["seed"] != row["seed"] or pair[2] != row["joint_id"] or pair[3] != row["state_values"]:
                records.append({**record, "images_match": False, "reason": "case_not_reproduced"})
                continue
            ref, chosen, _, _ = pair
            new = chosen["program"].spec.joints[0]
            expected = [sha256_path(out / "frames" / cid / f"{i}.png") for i in range(4)]
            observed = rendered_hashes(ref, chosen["render"], chosen["view"])
            match = expected == observed
            record |= {"new_candidate_sha": chosen["program"].sha, "new_sign": new.sign, "new_axis": list(new.axis),
                       "committed_image_sha256": expected, "rerendered_image_sha256": observed,
                       "images_match": match, "view": chosen["view"]}
            records.append(record)
            if match:
                (specs / f"{cid}.json").write_text(json.dumps(to_dict(chosen["program"].spec)), encoding="utf-8")
                updated[cid] = chosen
    for g in gold:
        if g["case_id"] in updated:
            chosen = updated[g["case_id"]]
            g["candidate_sha"], g["parameters"] = chosen["program"].sha, chosen["label"].params
        if g["kind"] == "direction_flip":
            g["stratum"] = "axis_negation" if g["case_id"] in updated else "sign_negation"
        elif g["kind"] == "joint_axis":
            g["stratum"] = "axis_random_angle"
            g["training_eligibility"] = "superseded_code_shortcut"
    write_jsonl(out / "private/gold.jsonl", gold)
    report = {"samples": len(records), "images_match": sum(r["images_match"] for r in records),
              "mismatched": [r["case_id"] for r in records if not r["images_match"]], "records": records}
    (out / "direction_flip_regeneration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {k: report[k] for k in ("samples", "images_match", "mismatched")}


def case_id_v2(asset_id: str, seed) -> str:
    return "eb2-" + hashlib.sha256(f"{SALT_V2}:{asset_id}:joint_axis:{seed}".encode()).hexdigest()[:12]


def build_axis_batch(src: Path, src_specs: Path, out: Path, specs: Path) -> dict:
    cfg, views = session_context()
    view_ids = [str(v.id) for v in views]
    view_by_id = {str(v.id): v for v in views}
    gold = v1.rows(src / "private/gold.jsonl")
    clean = sorted((g for g in gold if g["kind"] == "clean"), key=lambda g: g["asset_id"])
    old_axis = {g["asset_id"]: g["case_id"] for g in gold if g["kind"] == "joint_axis"}
    v1_view = {r["case_id"]: r["view"] for r in v1.rows(src / "review_manifest.jsonl")}
    category = {r["case_id"]: r["category"] for r in v1.rows(src / "review_manifest.jsonl")}
    for sub in ("private", "boards", "inputs", "frames"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    specs.mkdir(parents=True, exist_ok=True)
    done = {r["asset_id"] for r in v1.rows(out / "private/screening.jsonl")}
    with ObservationSession(cfg.render, views) as session:
        for index, g in enumerate(clean):
            asset_id = g["asset_id"]
            if asset_id in done:
                continue
            reference = ObjectProgram.from_spec(load_spec(src_specs / f"{asset_id}.reference.json"))
            if reference.sha != g["reference_sha"]:
                raise ValueError(f"reference of {asset_id} no longer matches the committed sha")
            record = {"asset_id": asset_id, "category": category[g["case_id"]], "clean_case_id": g["case_id"],
                      "had_random_angle_sample": asset_id in old_axis}
            if not get_corruption("kinematics.joint_axis", mode="axis_permutation").applicable(reference):
                v1.append(out / "private/screening.jsonl", {**record, "status": "skipped",
                          "reason": "reference_axis_not_a_coordinate_axis",
                          "reference_axes": [list(j.axis) for j in reference.spec.joints]})
                print(json.dumps({"asset": asset_id, "status": "skipped"}), flush=True)
                continue
            pair, diagnostics = find_case(session, reference, views, {"defect": "kinematics.joint_axis", "handler": HANDLER},
                                          SEED_BASE_V2 + index * 100_000)
            if not pair:
                v1.append(out / "private/screening.jsonl", {**record, "status": "rejected",
                          "reason": "no_permutation_passes_pixel_gates", "diagnostics": diagnostics})
                print(json.dumps({"asset": asset_id, "status": "rejected"}), flush=True)
                continue
            ref, chosen, joint_id, values = pair
            view, label, program = chosen["view"], chosen["label"], chosen["program"]
            same_reference = committed_hashes(src, g["case_id"])[:2] == [
                sha256_bytes(ref["frames"][s][view]) for s in ("rest", "limit")]
            if view != v1_view[g["case_id"]] or values != g["state_values"] or not same_reference:
                raise ValueError(f"{asset_id}: reference camera/state/images differ from easy-batch-v1")
            cid = case_id_v2(asset_id, chosen["seed"])
            check = checker_result(program, chosen["render"], reference, ref)
            if check["code_correct"]:
                raise ValueError(f"checker disagreement: {cid}")
            board(out / "boards" / f"{cid}.png", cid, ref, chosen["render"], view)
            pair_preview(out / "inputs" / f"{cid}.png", ref, chosen["render"], view)
            write_frames(out, cid, ref, chosen["render"], view)
            everywhere = render_states(session, program, chosen["framing"], joint_id, values, view_ids)
            v1.append(out / "private/view_visibility.jsonl", {
                "case_id": cid, "default_view": view,
                "per_view": per_view_visibility(everywhere, ref, "limit", chosen["framing"], view_by_id)})
            v1.append(out / "review_manifest.jsonl", {
                "case_id": cid, "category": record["category"], "view": view, "states": list(STATES),
                "input_states": ["rest", "limit"], "image_size": [256, 256], "board": f"boards/{cid}.png",
                "input_preview": f"inputs/{cid}.png", "images": [f"frames/{cid}/{i}.png" for i in range(4)],
                "status": "pending_human_review"})
            v1.append(out / "private/gold.jsonl", {
                **{k: g[k] for k in ("asset_id", "model_id", "original_spec_sha256", "reference_sha",
                                     "source_archive_sha256", "source_identity")},
                "case_id": cid, "difficulty": "easy_provisional", "kind": "joint_axis", "stratum": "axis_permutation",
                "expected_status": "fail", "defect_id": label.defect_id, "joint_id": joint_id,
                "parts": list(label.parts), "parameters": label.params, "seed": chosen["seed"],
                "state_values": values, "candidate_sha": program.sha, "clean_case_id": g["case_id"],
                "supersedes": old_axis.get(asset_id), "training_eligibility": "pending_human_review"})
            (specs / f"{cid}.json").write_text(json.dumps(to_dict(program.spec)), encoding="utf-8")
            v1.append(out / "private/screening.jsonl", {**record, "status": "automatic_pass", "case_id": cid,
                      "diagnostics": diagnostics})
            print(json.dumps({"asset": asset_id, "status": "automatic_pass", "case": cid,
                              "to": [label.params["to_index"], label.params["to_sign"]]}), flush=True)
    return finalize_axis_batch(out, specs, src)


def finalize_axis_batch(out: Path, specs: Path, src: Path) -> dict:
    gold = v1.rows(out / "private/gold.jsonl")
    screening = v1.rows(out / "private/screening.jsonl")
    public = sorted(v1.rows(out / "review_manifest.jsonl"), key=lambda r: r["case_id"])
    write_jsonl(out / "review_manifest.jsonl", public)
    reviews = {r["case_id"]: r for r in v1.rows(out / "human_review.jsonl")}
    template = {"status": "pending_human_review", "reviewer": None, "reviewed_at": None,
                "human_judgment": {"verdict": None, "one_glance": None, "what_is_wrong": None,
                                   "no_clipping_or_occlusion": None, "notes": None}}
    write_jsonl(out / "human_review.jsonl", [reviews.get(r["case_id"], {"case_id": r["case_id"], **template})
                                             for r in public])
    defects = sorted(gold, key=lambda g: hashlib.sha256(f"{SALT_V2}:spot:{g['case_id']}".encode()).hexdigest())[:SPOT_CHECK]
    cleans = sorted({g["clean_case_id"] for g in defects})
    (out / "spot_check.json").write_text(json.dumps({
        "rule": f"first {SPOT_CHECK} defect samples by sha256('{SALT_V2}:spot:' + case_id), plus their clean controls "
                "(same rule as easy-batch-v1). Clean controls are the easy-batch-v1 samples: their boards live in "
                "../easy-batch-v1/boards and their reviews belong in easy-batch-v1/human_review.jsonl",
        "case_ids": sorted(g["case_id"] for g in defects), "clean_control_case_ids": cleans}, indent=2), encoding="utf-8")
    cards = "\n".join(f'<section><h3>{c}</h3><img src="boards/{c}.png" width="100%"></section>'
                      for c in sorted(g["case_id"] for g in defects))
    cards += "\n" + "\n".join(f'<section><h3>{c}</h3><img src="../easy-batch-v1/boards/{c}.png" width="100%"></section>'
                               for c in cleans)
    (out / "review.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Easy batch v2 spot check</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto}}section{{margin:2em 0;border-bottom:1px solid #ccc}}</style>
<h1>Easy v2 joint_axis 抽查（盲审）</h1><p>上排参考、下排候选，列依次为 rest / mid / limit，同一固定相机。先不要打开 private/。后半部分是对应的 clean 对照（easy-batch-v1 的样本）。</p>{cards}""",
                                    encoding="utf-8")
    passed = [s for s in screening if s["status"] == "automatic_pass"]
    validation = {
        "status": "automatic_checks_passed", "review_status": "pending_human_review",
        "objects_considered": len(screening), "objects_accepted": len(passed),
        "objects_skipped_axis_not_coordinate": sum(s["status"] == "skipped" for s in screening),
        "objects_rejected_by_pixel_gates": sum(s["status"] == "rejected" for s in screening),
        "samples": len(gold), "by_stratum": dict(Counter(g["stratum"] for g in gold)),
        "supersedes_v1_samples": sum(g["supersedes"] is not None for g in gold),
        "objects_without_v1_sample_now_included": sum(g["supersedes"] is None for g in gold),
        "v1_objects_without_v2_sample": sorted(s["asset_id"] for s in screening
                                               if s["status"] != "automatic_pass" and s["had_random_angle_sample"]),
        "permutation_targets": dict(Counter(f"{g['parameters']['from_index']}->{g['parameters']['to_index']}"
                                            f"{'+' if g['parameters']['to_sign'] > 0 else '-'}" for g in gold)),
        "model_calls": 0}
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    refresh_manifest(out, specs, (Path(__file__).resolve(),))
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    flip = sub.add_parser("flip")
    flip.add_argument("--out", type=Path, default=v1.OUT)
    flip.add_argument("--specs", type=Path, default=v1.SPECS)
    axis = sub.add_parser("axis")
    axis.add_argument("--src", type=Path, default=v1.OUT)
    axis.add_argument("--src-specs", type=Path, default=v1.SPECS)
    axis.add_argument("--out", type=Path, default=OUT_V2)
    axis.add_argument("--specs", type=Path, default=SPECS_V2)
    args = parser.parse_args()
    if args.command == "flip":
        print(json.dumps(regenerate_direction_flip(args.out.resolve(), args.specs.resolve())))
    else:
        print(json.dumps(build_axis_batch(args.src.resolve(), args.src_specs.resolve(), args.out.resolve(),
                                          args.specs.resolve())))
