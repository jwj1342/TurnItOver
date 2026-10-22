#!/usr/bin/env python3
"""Batch construction of Easy samples (issue #9): single-joint objects, clean + joint_axis / direction_flip.

Same gates and frozen-camera protocol as ``construction_pilot``. Every row is ``pending_human_review``; the batch
is only spot-checked (``spot_check.json``), which is not a full human review. Resumable: objects already listed in
``private/screening.jsonl`` are skipped. Candidate specs are large and written outside Git to ``--specs``.
No model calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.audit_pilot_view_visibility import per_view_visibility
from scripts.construction_pilot import SALT, _hash, find_case, write_frames
from scripts.level0_expansion_search import render_states
from scripts.level0_expansion_sources import load_pool
from scripts.prepare_level0_axis_review import board, pair_preview
from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, ROOT, STATES, checker_result, sha256_path
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.urdf import UrdfModel
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

OUT = ROOT / "docs/evidence/easy-batch-v1"
SPECS = ROOT / "data/benchmarks/easy-batch-v1-specs"
SOURCES = (
    (ROOT / "data/assets/partnet-mobility-level0-round3/catalog.json", ROOT / "data/partnet-mobility-level0-round3/download-manifest.json"),
    (ROOT / "data/assets/partnet-mobility-level0-round2/catalog.json", ROOT / "data/partnet-mobility-level0-round2/download-manifest.json"),
)
DEFECTS = ("kinematics.joint_axis", "kinematics.direction_flip")
PRIOR_EVIDENCE = (ROOT / "docs/evidence/partnet-mobility-level0-axis-v4", ROOT / "docs/evidence/construction-pilot-v1")
SPOT_CHECK = 10
MAX_PER_CATEGORY = 8  # accepted objects per category, to keep the batch from collapsing onto one class


def prior_objects() -> set[str]:
    used = set()
    for bundle in PRIOR_EVIDENCE:
        used |= {json.loads(l)["asset_id"] for l in (bundle / "private/gold.jsonl").read_text().splitlines()}
        for name in ("screening.json", "expansion_screening.json", "inherited_screening.json"):
            path = bundle / "private" / name
            if path.exists():
                used |= {a["asset_id"] for a in json.loads(path.read_text())["assets"]}
    return used


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def case_id(asset_id: str, kind: str, seed) -> str:
    return f"eb-{_hash('easy-batch', asset_id, kind, seed)[:12]}"


def build(out: Path, specs: Path, target: int, max_tries: int) -> dict:
    cfg = load_generate_config(DEFAULT_CONFIG)
    if (cfg.render.width, cfg.render.height) != (256, 256):
        raise ValueError("batch requires native 256x256 rendering")
    views = load_views(cfg.views_path)
    view_ids = [str(v.id) for v in views]
    view_by_id = {str(v.id): v for v in views}
    excluded = prior_objects()
    pool = []
    for catalog, download in SOURCES:
        pool += [r for r in load_pool(catalog_path=catalog, download_path=download)[0]
                 if len(r["spec"].joints) == 1 and r["spec"].asset_id not in excluded]
    pool.sort(key=lambda r: _hash("easy-order", r["spec"].asset_id))
    for sub in ("private", "boards", "inputs", "frames"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    specs.mkdir(parents=True, exist_ok=True)
    done = {r["asset_id"] for r in rows(out / "private/screening.jsonl")}
    accepted = sum(r["status"] == "automatic_pass" for r in rows(out / "private/screening.jsonl"))
    categories = Counter(s["category"] for s in rows(out / "private/screening.jsonl") if s["status"] == "automatic_pass")
    with ObservationSession(cfg.render, views) as session:
        for index, item in enumerate(pool):
            spec, meta = item["spec"], item["metadata"]
            if accepted >= target or len(done) >= max_tries:
                break
            if spec.asset_id in done or categories[spec.category] >= MAX_PER_CATEGORY:
                continue
            reference = ObjectProgram.from_spec(spec)
            found, diagnostics = {}, {}
            for defect in DEFECTS:
                pair, diagnostics[defect] = find_case(session, reference, views, {"defect": defect}, 3_000_000 + index * 100_000)
                if pair:
                    found[defect] = pair
            authenticity = None
            if found:
                try:
                    authenticity = audit_model(session, UrdfModel(Path(meta["source_urdf"])), spec)
                except Exception as error:  # keep the raw failure, never infer fidelity
                    authenticity = {"passed": False, "error_type": type(error).__name__, "error": str(error)}
                if authenticity.get("passed") is not True:
                    found = {}
            status = "automatic_pass" if found else "rejected"
            done.add(spec.asset_id)
            if found:
                accepted += 1
                categories[spec.category] += 1
                ref, first, joint_id, values = next(iter(found.values()))
                view, framing = first["view"], first["framing"]
                if any(p[1]["view"] != view or p[1]["framing"] != framing or p[2] != joint_id for p in found.values()):
                    raise ValueError(f"defects of {spec.asset_id} did not share one frozen camera and joint")
                variants = [("clean", None, reference, ref, None)] + [
                    (d.split(".")[1], p[1]["seed"], p[1]["program"], p[1]["render"], p[1]["label"]) for d, p in found.items()]
                (specs / f"{spec.asset_id}.reference.json").write_text(json.dumps(to_dict(spec)), encoding="utf-8")
                for kind, seed, program, rendered, label in variants:
                    cid = case_id(spec.asset_id, kind, seed)
                    check = checker_result(program, rendered, reference, ref)
                    if check["code_correct"] != (label is None):
                        raise ValueError(f"checker disagreement: {cid}")
                    board(out / "boards" / f"{cid}.png", cid, ref, rendered, view)
                    pair_preview(out / "inputs" / f"{cid}.png", ref, rendered, view)
                    write_frames(out, cid, ref, rendered, view)
                    visibility = None
                    if label is not None:
                        everywhere = render_states(session, program, framing, joint_id, values, view_ids)
                        visibility = per_view_visibility(everywhere, ref, "limit", framing, view_by_id)
                    append(out / "review_manifest.jsonl", {
                        "case_id": cid, "category": spec.category, "view": view, "states": list(STATES),
                        "input_states": ["rest", "limit"], "image_size": [256, 256],
                        "board": f"boards/{cid}.png", "input_preview": f"inputs/{cid}.png",
                        "images": [f"frames/{cid}/{i}.png" for i in range(4)], "status": "pending_human_review"})
                    append(out / "private/gold.jsonl", {
                        "case_id": cid, "asset_id": spec.asset_id, "difficulty": "easy_provisional",
                        "kind": "clean" if label is None else kind, "expected_status": "pass" if label is None else "fail",
                        "defect_id": None if label is None else label.defect_id,
                        "joint_id": None if label is None else joint_id, "parts": [] if label is None else list(label.parts),
                        "parameters": {} if label is None else label.params, "seed": seed, "state_values": values,
                        "candidate_sha": program.sha, "reference_sha": reference.sha,
                        "original_spec_sha256": meta["original_spec_sha256"], "model_id": meta["model_id"],
                        "source_archive_sha256": meta["archive_sha256"], "source_identity": meta["source_identity"],
                        "training_eligibility": "pending_human_review"})
                    if visibility is not None:
                        append(out / "private/view_visibility.jsonl", {"case_id": cid, "default_view": view,
                                                                         "per_view": visibility})
                    (specs / f"{cid}.json").write_text(json.dumps(to_dict(program.spec)), encoding="utf-8")
            append(out / "private/screening.jsonl", {
                "asset_id": spec.asset_id, "category": spec.category, "status": status,
                "defects_passed": sorted(found), "source_authenticity": authenticity, "diagnostics": diagnostics})
            print(json.dumps({"asset": spec.asset_id, "category": spec.category, "pass": sorted(found),
                              "accepted": accepted, "tried": len(done)}), flush=True)
    return finalize(out, specs, target)


def finalize(out: Path, specs: Path, target: int) -> dict:
    gold = rows(out / "private/gold.jsonl")
    screening = rows(out / "private/screening.jsonl")
    public = sorted(rows(out / "review_manifest.jsonl"), key=lambda r: r["case_id"])
    (out / "review_manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in public), encoding="utf-8")
    reviews = {r["case_id"]: r for r in rows(out / "human_review.jsonl")}
    template = {"status": "pending_human_review", "reviewer": None, "reviewed_at": None,
                "human_judgment": {"verdict": None, "one_glance": None, "what_is_wrong": None,
                                   "no_clipping_or_occlusion": None, "notes": None}}
    (out / "human_review.jsonl").write_text("".join(
        json.dumps(reviews.get(r["case_id"], {"case_id": r["case_id"], **template}), ensure_ascii=False, sort_keys=True) + "\n"
        for r in public), encoding="utf-8")
    # Deterministic spot check: a hash-ordered sample over defect samples plus their clean controls' objects.
    defects = sorted((g for g in gold if g["kind"] != "clean"), key=lambda g: _hash("spot", g["case_id"]))[:SPOT_CHECK]
    spot = sorted({g["case_id"] for g in defects} | {next(c["case_id"] for c in gold if c["asset_id"] == g["asset_id"] and c["kind"] == "clean")
                                                     for g in defects})
    (out / "spot_check.json").write_text(json.dumps({"rule": f"first {SPOT_CHECK} defect samples by sha256('{SALT}:spot:' + case_id), plus their clean controls",
                                                    "case_ids": spot}, indent=2), encoding="utf-8")
    cards = "\n".join(f'<section><h3>{c}</h3><img src="boards/{c}.png" width="100%"></section>' for c in spot)
    (out / "review.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Easy batch spot check</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto}}section{{margin:2em 0;border-bottom:1px solid #ccc}}</style>
<h1>Easy 批次抽查（盲审）</h1><p>上排参考、下排候选，列依次为 rest / mid / limit，同一固定相机。先不要打开 private/。</p>{cards}""", encoding="utf-8")
    objects = [s for s in screening if s["status"] == "automatic_pass"]
    reasons = Counter(r for s in screening if s["status"] == "rejected" for d in s["diagnostics"].values()
                      for j in d for r in (j.get("rejection_reasons", []) + [x for a in j["attempts"] for x in a["rejection_reasons"]]))
    validation = {"status": "automatic_checks_passed", "review_status": "spot_check_pending", "target_objects": target,
                  "objects_tried": len(screening), "objects_accepted": len(objects),
                  "samples": len(public), "clean": sum(g["kind"] == "clean" for g in gold),
                  "by_defect": dict(Counter(g["defect_id"] for g in gold if g["defect_id"])),
                  "objects_by_category": dict(Counter(o["category"] for o in objects)),
                  "rejection_reason_counts_upper_bound": dict(reasons.most_common(8)), "model_calls": 0}
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    manifest = {"version": 1, "inputs": {str(Path(__file__).resolve()): sha256_path(Path(__file__).resolve())},
                "files": {p.relative_to(out).as_posix(): sha256_path(p) for p in sorted(out.rglob("*"))
                          if p.is_file() and p.name not in ("manifest.json", "human_review.jsonl")},
                "specs_outside_git": str(specs),
                "spec_sha256": {p.name: sha256_path(p) for p in sorted(specs.glob("*.json"))},
                "note": "human_review.jsonl is the only file edited after generation and is not hash-pinned."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--specs", type=Path, default=SPECS)
    parser.add_argument("--target", type=int, default=50, help="new Easy objects to accept")
    parser.add_argument("--max-tries", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(build(args.out.resolve(), args.specs.resolve(), args.target, args.max_tries)))
