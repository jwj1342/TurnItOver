"""Freeze and load the Benchmark v1 manifest, split and gold files.

``build_artifacts`` is the only place that reads the Level 0 evidence package; everything downstream
(loader, scorer, audit) reads the frozen files under ``benchmark/v1`` and re-verifies their SHA-256.
Split assignment uses object identity and a fixed salt only; model results are never an input.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from turnitover.assets.emit import emit_program
from turnitover.benchmark.protocol import (BENCHMARK_VERSION, DIFFICULTIES, IMAGE_ROLES,
                                          condition_inputs,  # noqa: F401  re-exported for existing callers
                                          protocol_document)
from turnitover.core.serde import from_dict
from turnitover.core.spec import AssetSpec

EVIDENCE = "docs/evidence/partnet-mobility-level0-axis-v4"
EXPOSED_SOURCE = "docs/evidence/partnet-mobility-level0-axis-v2/object_splits.json"
OUT = "benchmark/v1"
FILES = ("manifest.jsonl", "gold.jsonl", "split.json", "protocol.json")
SPLIT_SALT = "turnitover-benchmark-v1-split"
VALIDATION_FRACTION = 0.2
PROMPT = ("Images: reference rest, reference limit, candidate rest, candidate limit. "
          "Does the candidate follow the reference motion? Decide pass, fail, uncertain or continue.")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _dump_jsonl(rows) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def difficulty_of(n_joints: int) -> str:
    """Level 0 only injects one joint_axis error: single-joint objects are easy, the rest medium."""
    return "easy" if n_joints <= DIFFICULTIES["easy"]["max_joints"] else "medium"


def assign_validation(objects: dict[str, dict]) -> set[str]:
    """Pick ~20% of each tier for Validation, only among objects unseen by earlier experiments."""
    validation: set[str] = set()
    for tier in DIFFICULTIES:
        members = [a for a, o in objects.items() if o["difficulty"] == tier]
        fresh = sorted((a for a in members if not objects[a]["exposed_to_prior_experiments"]),
                       key=lambda a: hashlib.sha256(f"{SPLIT_SALT}:{a}".encode()).hexdigest())
        validation.update(fresh[:min(round(VALIDATION_FRACTION * len(members)), len(fresh))])
    return validation


def _geometry_fingerprint(spec: dict) -> str:
    parts = sorted(json.dumps([p.get("mesh"), p.get("size"), p.get("origin"), p.get("rotation")],
                              sort_keys=True) for p in spec["parts"])
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _state_values(root: Path, evidence: Path) -> dict[tuple[str, str], dict]:
    """Recorded rest/mid/limit values per (asset, joint); only automatic-pass records are trusted."""
    found = {}
    for name in ("screening.json", "expansion_screening.json"):
        for asset in json.loads((evidence / "private" / name).read_text(encoding="utf-8"))["assets"]:
            for joint in asset.get("joints", []):
                if asset["status"] == "automatic_pass" and any(a.get("automatic_pass") for a in joint.get("attempts", [])):
                    found[(asset["asset_id"], joint["joint_id"])] = joint["state_values"]
    return found


def build_artifacts(root: Path) -> dict[str, str]:
    evidence = root / EVIDENCE
    gold = {g["case_id"]: g for g in read_jsonl(evidence / "private" / "gold.jsonl")}
    inputs = {r["case_id"]: r for r in read_jsonl(evidence / "train_manifest.jsonl")}
    reviews = {r["case_id"]: r for r in read_jsonl(evidence / "human_review.jsonl")}
    object_info = {o["asset_id"]: o for o in json.loads((evidence / "object_splits.json").read_text(encoding="utf-8"))["objects"]}
    exposed = {o["asset_id"] for o in json.loads((root / EXPOSED_SOURCE).read_text(encoding="utf-8"))["objects"]}
    inherited = {a["asset_id"]: a for a in json.loads((evidence / "private" / "inherited_screening.json").read_text(encoding="utf-8"))["assets"]}
    states = _state_values(root, evidence)
    if set(gold) != set(inputs) or set(gold) != set(reviews):
        raise ValueError("gold, train_manifest and human_review case ids differ")

    specs = {cid: json.loads((evidence / "private" / f"{cid}.json").read_text(encoding="utf-8")) for cid in gold}
    clean_case = {g["asset_id"]: cid for cid, g in gold.items() if g["kind"] == "clean"}
    axis_joint = {g["asset_id"]: g["joint_id"] for g in gold.values() if g["kind"] != "clean"}
    objects = {}
    for asset_id, info in sorted(object_info.items()):
        spec = specs[clean_case[asset_id]]
        objects[asset_id] = {
            "asset_id": asset_id, "split_group": info["split_group"], "category": spec["category"],
            "n_joints": len(spec["joints"]), "difficulty": difficulty_of(len(spec["joints"])),
            "exposed_to_prior_experiments": asset_id in exposed,
            "source_model_id": info["source_model_id"], "source_archive_sha256": info["source_archive_sha256"],
            "geometry_fingerprint": _geometry_fingerprint(spec),
            "reference_program_sha": gold[clean_case[asset_id]]["reference_sha"]}
    validation = assign_validation(objects)
    for asset_id, o in objects.items():
        o["split"] = "validation" if asset_id in validation else "train"

    manifest, gold_rows = [], []
    for cid in sorted(gold):
        g, o = gold[cid], objects[gold[cid]["asset_id"]]
        review = reviews[cid]
        if review["status"] != "human_review_complete" or review["human_judgment"]["verdict"] != g["expected_status"]:
            raise ValueError(f"review and gold disagree for {cid}")
        joint_id = axis_joint[o["asset_id"]]
        values = states.get((o["asset_id"], joint_id))
        joint = next(j for j in specs[cid]["joints"] if j["id"] == joint_id)
        images = []
        for index, role in enumerate(IMAGE_ROLES):
            rel = inputs[cid]["images"][index]
            images.append({"index": index, "role": role, "path": f"{EVIDENCE}/{rel}", "width": 256, "height": 256,
                           "sha256": sha256_file(evidence / rel)})
        ref_cid = clean_case[o["asset_id"]]
        manifest.append({
            "sample_id": cid, "benchmark": BENCHMARK_VERSION, "split": o["split"], "difficulty": o["difficulty"],
            "asset_id": o["asset_id"], "split_group": o["split_group"], "category": o["category"],
            "prompt": PROMPT, "images": images,
            "context": {"joint_ids": sorted(j["id"] for j in specs[cid]["joints"]),
                        "part_ids": sorted(p["id"] for p in specs[cid]["parts"])},
            "code": [{"role": "reference_spec", "path": f"{EVIDENCE}/private/{ref_cid}.json",
                      "sha256": sha256_file(evidence / "private" / f"{ref_cid}.json")},
                     {"role": "candidate_spec", "path": f"{EVIDENCE}/private/{cid}.json",
                      "sha256": sha256_file(evidence / "private" / f"{cid}.json")}],
            "frozen_state": {
                "joint_id": joint_id, "rest": 0.0,
                "limit": values["limit"] if values else joint["limits"][1],
                "mid": values["mid"] if values else joint["limits"][1] / 2,
                "source": "screening_record" if values else "detent_rule_from_spec_limits"},
            "camera": {"view_id": inherited.get(o["asset_id"], {}).get("view"),
                       "view_recorded": o["asset_id"] in inherited,
                       "note": "Camera is frozen by the image SHA-256; the per-sample view id was not recorded "
                               "in the source package unless view_recorded is true."}})
        gold_rows.append({
            "sample_id": cid, "asset_id": o["asset_id"], "difficulty": o["difficulty"], "split": o["split"],
            "status": g["expected_status"], "n_errors": 0 if g["kind"] == "clean" else 1,
            "defects": [] if g["kind"] == "clean" else [
                {"defect_id": g["defect_id"], "joint_id": g["joint_id"], "parts": sorted(g["parts"])}],
            "candidate_program_sha": g["candidate_sha"], "reference_program_sha": g["reference_sha"],
            "review": {"status": review["status"], "mode": review.get("review_mode", "unspecified_in_source"),
                       "reviewed_at": review.get("reviewed_at")}})
    split = {"version": 1, "benchmark": BENCHMARK_VERSION, "unit": "independent_source_object",
             "rule": (f"Per difficulty tier, round({VALIDATION_FRACTION} * tier objects) Validation objects taken in "
                      f"ascending sha256('{SPLIT_SALT}:' + asset_id) order from objects not exposed to prior "
                      "experiments; everything else is Train. No model output is an input."),
             "exposure_source": EXPOSED_SOURCE, "evidence_package": EVIDENCE,
             "objects": [objects[a] for a in sorted(objects)]}
    return {"manifest.jsonl": _dump_jsonl(manifest), "gold.jsonl": _dump_jsonl(gold_rows),
            "split.json": _dump(split), "protocol.json": _dump(protocol_document())}


def freeze(root: Path, out: Path | None = None) -> dict:
    out = out or root / OUT
    if (out / "FREEZE.json").exists():
        raise FileExistsError(f"{out} is already frozen; create a new version instead of overwriting")
    out.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifacts(root)
    for name, text in artifacts.items():
        (out / name).write_text(text, encoding="utf-8")
    freeze_doc = {"benchmark": BENCHMARK_VERSION, "files": {n: sha256_file(out / n) for n in FILES}}
    (out / "FREEZE.json").write_text(_dump(freeze_doc), encoding="utf-8")
    return freeze_doc


def verify_freeze(bench_dir: Path) -> dict:
    doc = json.loads((bench_dir / "FREEZE.json").read_text(encoding="utf-8"))
    for name in FILES:
        if sha256_file(bench_dir / name) != doc["files"][name]:
            raise ValueError(f"{name} does not match FREEZE.json")
    return doc


def load_manifest(bench_dir: Path, split: str | None = None) -> list[dict]:
    verify_freeze(bench_dir)
    rows = read_jsonl(bench_dir / "manifest.jsonl")
    return [r for r in rows if split is None or r["split"] == split]


def load_gold(bench_dir: Path, split: str | None = None) -> dict[str, dict]:
    verify_freeze(bench_dir)
    return {r["sample_id"]: r for r in read_jsonl(bench_dir / "gold.jsonl") if split is None or r["split"] == split}


def load_samples(root: Path, bench_dir: Path, split: str | None = None) -> list[dict]:
    """Return manifest rows whose images were actually decoded and SHA-checked on disk."""
    from PIL import Image
    rows = load_manifest(bench_dir, split)
    for row in rows:
        for image in row["images"]:
            path = root / image["path"]
            if sha256_file(path) != image["sha256"]:
                raise ValueError(f"image SHA mismatch: {image['path']}")
            with Image.open(path) as handle:
                handle.load()
                if handle.size != (image["width"], image["height"]):
                    raise ValueError(f"image size mismatch: {image['path']}")
        for code in row["code"]:
            if sha256_file(root / code["path"]) != code["sha256"]:
                raise ValueError(f"code SHA mismatch: {code['path']}")
    return rows


def candidate_program_source(spec_path: Path) -> tuple[str, str]:
    """TypeScript of the candidate program (``emit_program`` of its spec) and its SHA-256; the v2 ``code_read`` input."""
    source = emit_program(from_dict(AssetSpec, json.loads(spec_path.read_text(encoding="utf-8"))))
    return source, hashlib.sha256(source.encode("utf-8")).hexdigest()
