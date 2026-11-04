"""Collect v1.1 same-moving-part two-field Hard candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from scripts.hard_construction.build_hard_full_motion_examples import view_pair
from scripts.hard_construction.hard_compositional_policy import POLICY, SIGNATURES, apply
from scripts.hard_construction.hard_geometry import normal_framing, spec_diagonal
from scripts.tier_construction.level0_expansion_search import inside_camera, render_states
from scripts.tier_construction.prepare_obvious_defect_review import DEFAULT_CONFIG, checker_result, sha256_path
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.catalog import decode_spec
from turnitover.assets.urdf import UrdfModel
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

STATES = ("rest", "mid", "limit")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values(target: dict) -> dict:
    endpoint = max(target["original_limits"], key=abs)
    return {"rest": 0., "mid": endpoint / 2, "limit": endpoint}


def split(index: int) -> str:
    return "train" if index < 25 else "validation" if index < 45 else "test"


def save_case(out, cid, normal, rendered, views, reference, candidate, payload):
    files = {}
    with tempfile.TemporaryDirectory(prefix=".hard-v11-", dir=out.parent) as temp:
        stage = Path(temp)
        for view in views:
            for role, frames in (("reference", normal), ("candidate", rendered)):
                for state in STATES:
                    path = stage / f"native/{cid}/{view}/{role}_{state}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(frames["frames"][state][view])
        for name, program in ((f"private/programs/{cid}.json", candidate),):
            path = stage / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(to_dict(program.spec), ensure_ascii=False, indent=2) + "\n")
        for path in stage.rglob("*"):
            if path.is_file():
                destination = out / path.relative_to(stage); destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination); files[destination.relative_to(out).as_posix()] = sha(destination)
    payload["files"] = files


def collect(pool_path: Path, out: Path, target: int, limit: int) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be new")
    pool = json.loads(pool_path.read_text())
    cfg = load_generate_config(DEFAULT_CONFIG)
    state = {"version": 1, "policy": POLICY, "target_new_objects": target,
             "pool_sha256": sha(pool_path), "inputs": {str(pool_path.resolve()): sha(pool_path),
             str(Path(__file__).resolve()): sha(Path(__file__)),
             str(Path(__file__).with_name("hard_compositional_policy.py").resolve()): sha(Path(__file__).with_name("hard_compositional_policy.py")),
             str(Path(DEFAULT_CONFIG).resolve()): sha(Path(DEFAULT_CONFIG)), str(cfg.views_path.resolve()): sha(cfg.views_path)}}
    write(out / "private/run_state.json", state)
    views_tuple, view_map = load_views(cfg.views_path), None
    view_map = {str(view.id): view for view in views_tuple}
    accepted, rejected = [], []
    with ObservationSession(cfg.render, views_tuple) as session:
        for row in pool:
            if len(accepted) >= target or len(accepted) + len(rejected) >= limit:
                break
            source = json.loads(Path(row["source_record"]).read_text())
            spec = decode_spec(json.loads(Path(row["source_spec"]).read_text()))
            reference = ObjectProgram.from_spec(spec)
            target_row = row["available_targets"][0]
            signature = SIGNATURES[len(accepted) % len(SIGNATURES)]
            state_values = values(target_row)
            try:
                seed = render_states(session, reference, None, target_row["joint_id"], state_values, ["front"])
                framing = normal_framing(seed, 1.15)
                normal = render_states(session, reference, framing, target_row["joint_id"], state_values, list(view_map))
                initial, supplementary = view_pair(normal, framing, view_map, target_row["part_id"])
                candidate, defects = apply(reference, target_row["joint_id"], target_row["part_id"], signature,
                                           spec_diagonal(to_dict(spec)), len(accepted) + len(rejected))
                rendered = render_states(session, candidate, framing, target_row["joint_id"], state_values,
                                         [initial, supplementary])
                if any(not inside_camera(rendered, framing, view_map[view])[0] for view in (initial, supplementary)):
                    raise ValueError("candidate_clipped")
                check, clean = checker_result(candidate, rendered, reference, normal), checker_result(reference, normal, reference, normal)
                authenticity = audit_model(session, UrdfModel(Path(row["source_urdf"])), spec)
                if check["code_correct"] or not check["checks"][1]["passed"] or not clean["code_correct"] or not authenticity["passed"]:
                    raise ValueError("combined_checker_or_source_authenticity_failed")
            except (ValueError, KeyError) as error:
                rejected.append({"asset_id": row["asset_id"], "reason": str(error)}); continue
            cid = "hp-" + hashlib.sha256(f"{POLICY}:{row['asset_id']}:{signature}".encode()).hexdigest()[:14]
            source_meta = {**source["metadata"], "source_urdf": row["source_urdf"], "split": split(len(accepted)),
                           "complexity": {"active_joints": row["active_joints"], "parts": row["parts"], "category": row["category"]}}
            public = {"case_id": cid, "initial_view": initial, "supplementary_view": supplementary, "states": list(STATES),
                      "native_root": f"native/{cid}", "image_size": [256, 256], "status": "pending_human_review"}
            gold = {"case_id": cid, "asset_id": row["asset_id"], "construction_track": "compositional_v1_1",
                    "fault_count": 2, "fault_signature": signature, "defects": defects, "state_values": state_values,
                    "reference_sha": reference.sha, "candidate_sha": candidate.sha, "program": f"private/programs/{cid}.json",
                    "source": source_meta, "training_eligibility": "pending_human_review"}
            review = {"case_id": cid, "status": "pending_human_review", "reviewer": None, "reviewed_at": None,
                      "human_judgment": {"both_faults_identified": None, "notes": None}}
            record = {"status": "automatic_pass", "asset_id": row["asset_id"], "category": row["category"],
                      "public": public, "gold": gold, "checks": {"case_id": cid, **check, "clean_check": clean,
                      "source_authenticity": authenticity}, "review": review, "files": {}}
            save_case(out, cid, normal, rendered, (initial, supplementary), reference, candidate, record)
            write(out / f"private/objects/{row['asset_id']}/record.json", record); accepted.append(record)
    for name, key in (("pool_manifest.jsonl", "public"), ("private/gold.jsonl", "gold"),
                      ("private/checks.jsonl", "checks"), ("human_review.jsonl", "review")):
        rows(out / name, [record[key] for record in accepted])
    rows(out / "train_manifest.jsonl", [])
    write(out / "private/rejections.json", rejected)
    report = {"status": "ready_pending_human_review" if len(accepted) == target else "collection_incomplete",
              "new_objects": len(accepted), "target_new_objects": target, "cases": len(accepted),
              "rejections": len(rejected), "training_eligible_cases": 0}
    write(out / "validation.json", report)
    write(out / "manifest.json", {**state, **report, "files": {p.relative_to(out).as_posix(): sha(p)
          for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=75); parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args(); print(json.dumps(collect(args.pool, args.out, args.target, args.limit), ensure_ascii=False))
