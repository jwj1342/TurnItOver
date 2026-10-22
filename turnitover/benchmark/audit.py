"""Benchmark v1 data audit: loader, leakage, difficulty statistics, scorer reproducibility and gaps.

``failures`` are violated invariants and make the audit fail. ``gaps`` are honest shortfalls against the
issue's four-tier target (missing tiers, defect types, evidence-insufficient samples) and do not.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

from turnitover.benchmark.manifest import (EVIDENCE, load_gold, load_manifest, load_samples, sha256_file,
                                           verify_freeze, read_jsonl)
from turnitover.benchmark.protocol import CONDITIONS, DIFFICULTIES, protocol_document
from turnitover.benchmark.scoring import score

PUBLIC_FORBIDDEN = {"status", "defects", "n_errors", "review", "candidate_program_sha", "reference_program_sha"}
OBJECT_KEYS = ("asset_id", "split_group", "source_model_id", "source_archive_sha256",
               "geometry_fingerprint", "reference_program_sha")
DEFERRED_TIERS = ("hard", "expert")
AGGREGATE_MODE = "user_reported_visual_review_aggregate_confirmation"


def _table(gold_rows) -> dict:
    """Table 1: objects and mutually exclusive sample counts per difficulty."""
    table = {}
    for tier in DIFFICULTIES:
        rows = [g for g in gold_rows if g["difficulty"] == tier]
        kinds = Counter("clean" if not g["defects"] else "compound" if len(g["defects"]) > 1
                        else g["defects"][0]["defect_id"] for g in rows)
        table[tier] = {"objects": len({g["asset_id"] for g in rows}), "clean": kinds["clean"],
                       "joint_axis": kinds["kinematics.joint_axis"], "part_offset": kinds["structure.part_offset"],
                       "compound": kinds["compound"], "total": len(rows)}
    table["total"] = {k: sum(table[t][k] for t in DIFFICULTIES) if k != "objects"
                      else len({g["asset_id"] for g in gold_rows}) for k in table["easy"]}
    return table


def _scorer_reproducibility(bench_dir: Path) -> dict:
    gold = load_gold(bench_dir, "validation")
    oracle = [{"case_id": s, "decision": "pass" if g["status"] == "pass" else "fail",
               "findings": [{"defect_id": d["defect_id"], "joint_id": d["joint_id"], "parts": d["parts"],
                             "evidence": [1, 3]} for d in g["defects"]]} for s, g in sorted(gold.items())]
    abstain = [{"case_id": s, "decision": "uncertain", "findings": []} for s in sorted(gold)]
    result = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, rows, condition in (("oracle", oracle, "rgb"), ("abstain", abstain, "no_image")):
            predictions = tmp / f"{name}.jsonl"
            predictions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            first = score(bench_dir, predictions, tmp / f"{name}-1.json", condition=condition)
            score(bench_dir, predictions, tmp / f"{name}-2.json", condition=condition)
            same = sha256_file(tmp / f"{name}-1.json") == sha256_file(tmp / f"{name}-2.json")
            result[name] = {"condition": condition, "byte_identical_rerun": same, "metrics": first["metrics"]}
    return result


def audit(root: Path, bench_dir: Path) -> dict:
    failures: list[str] = []
    freeze = verify_freeze(bench_dir)
    if json.loads((bench_dir / "protocol.json").read_text(encoding="utf-8")) != json.loads(
            json.dumps(protocol_document())):
        failures.append("protocol.json differs from turnitover.benchmark.protocol")
    samples = load_samples(root, bench_dir)
    manifest = load_manifest(bench_dir)
    gold = load_gold(bench_dir)
    split = json.loads((bench_dir / "split.json").read_text(encoding="utf-8"))
    objects = {o["asset_id"]: o for o in split["objects"]}

    per_split = Counter(r["split"] for r in samples)
    for name in ("train", "validation"):
        if per_split[name] == 0:
            failures.append(f"loader read zero samples for split {name}")
    if set(manifest_row["sample_id"] for manifest_row in manifest) != set(gold):
        failures.append("manifest and gold sample ids differ")

    # Object-level pairing: exactly one clean and one defect sample, both in the same split.
    by_object = defaultdict(list)
    for r in manifest:
        by_object[r["asset_id"]].append(r)
    for asset_id, rows in by_object.items():
        statuses = sorted(gold[r["sample_id"]]["status"] for r in rows)
        if statuses != ["fail", "pass"]:
            failures.append(f"{asset_id}: expected one clean and one defect sample, got {statuses}")
        if len({r["split"] for r in rows}) != 1 or rows[0]["split"] != objects[asset_id]["split"]:
            failures.append(f"{asset_id}: samples and split.json disagree on split")

    # Leakage: no object, source or image shared across Train and Validation.
    leaks = {}
    for key in OBJECT_KEYS:
        seen = defaultdict(set)
        for o in objects.values():
            seen[o[key]].add(o["asset_id"])
        shared_between_objects = {v: sorted(a) for v, a in seen.items() if len(a) > 1}
        across = {v: sorted(a) for v, a in shared_between_objects.items()
                  if len({objects[x]["split"] for x in a}) > 1}
        leaks[key] = {"values_shared_between_objects": len(shared_between_objects), "across_splits": len(across)}
        if across:
            failures.append(f"{key} shared across Train/Validation: {sorted(across)[:3]}")
        if key != "reference_program_sha" and shared_between_objects:
            failures.append(f"{key} not unique per object: {sorted(shared_between_objects)[:3]}")
    image_split = defaultdict(set)
    for r in manifest:
        for image in r["images"]:
            image_split[image["sha256"]].add(r["split"])
    cross_images = sorted(h for h, s in image_split.items() if len(s) > 1)
    if cross_images:
        failures.append(f"{len(cross_images)} identical image SHA-256 values appear in both splits")
    exposed_validation = sorted(a for a, o in objects.items()
                                if o["split"] == "validation" and o["exposed_to_prior_experiments"])
    if exposed_validation:
        failures.append(f"Validation contains objects used in prior experiments: {exposed_validation}")

    # Public inputs must not carry gold, and every condition may read only public fields.
    public_keys = set(manifest[0])
    for name, cond in CONDITIONS.items():
        bad = (set(cond["fields"]) & PUBLIC_FORBIDDEN) or (set(cond["fields"]) - public_keys)
        if bad:
            failures.append(f"condition {name} reads forbidden or unknown fields {sorted(bad)}")
    if PUBLIC_FORBIDDEN & public_keys:
        failures.append(f"manifest rows expose gold fields {sorted(PUBLIC_FORBIDDEN & public_keys)}")

    tables = {name: _table([g for g in gold.values() if name == "all" or g["split"] == name])
              for name in ("all", "train", "validation")}
    reproducibility = _scorer_reproducibility(bench_dir)
    for name, result in reproducibility.items():
        if not result["byte_identical_rerun"]:
            failures.append(f"scorer rerun is not byte-identical for {name}")
    if reproducibility["oracle"]["metrics"]["exact_verdict"] != 1.0:
        failures.append("oracle predictions do not score exact_verdict=1.0")

    reviews = read_jsonl(root / EVIDENCE / "human_review.jsonl")
    gaps = []
    for tier in DIFFICULTIES:
        for name in ("train", "validation"):
            if tables[name][tier]["total"] == 0:
                later = " (deferred: collected later)" if tier in DEFERRED_TIERS else ""
                gaps.append(f"{tier}: no {name} samples{later}")
    for defect in sorted({d for t in DIFFICULTIES.values() for d in t["defects"]}):
        key = {"kinematics.joint_axis": "joint_axis", "structure.part_offset": "part_offset"}[defect]
        if tables["all"]["total"][key] == 0:
            gaps.append(f"{defect}: no samples in any tier")
    if tables["all"]["total"]["compound"] == 0:
        gaps.append("compound errors: no samples (expert tier is empty, deferred: collected later)")
    if not any(g["status"] == "uncertain" for g in gold.values()):
        gaps.append("no gold-uncertain (evidence-insufficient) samples; abstention is only exercised by the "
                    "no_image condition")
    unrecorded = sum(1 for r in manifest if not r["camera"]["view_recorded"])
    if unrecorded:
        gaps.append(f"{unrecorded}/{len(manifest)} samples have no recorded camera view id (frozen via image SHA-256)")
    aggregate = sum(1 for r in reviews if r.get("review_mode") == AGGREGATE_MODE)
    if aggregate:
        gaps.append(f"{aggregate}/{len(reviews)} human reviews are aggregate user confirmations, not per-case labels")
    if per_split["validation"] < 40:
        gaps.append(f"validation has only {per_split['validation']} samples from "
                    f"{len({r['asset_id'] for r in samples if r['split'] == 'validation'})} objects; "
                    "metric confidence intervals will be wide")
    gaps.append("original 7 Validation groups (PartNet-Mobility 1280/10797/34610/45759/100202/102252/102590) "
                "have no Level 0 samples and are not part of Benchmark v1")

    return {"benchmark": freeze["benchmark"], "status": "failed" if failures else "passed",
            "failures": failures, "gaps": gaps, "freeze_files": freeze["files"],
            "loader": {"samples": len(samples), "per_split": dict(per_split),
                       "images_decoded_and_sha_verified": 4 * len(samples)},
            "leakage": leaks | {"identical_image_sha_across_splits": len(cross_images),
                                "exposed_objects_in_validation": len(exposed_validation)},
            "table1": tables,
            "split_objects": dict(Counter((o["difficulty"], o["split"], o["exposed_to_prior_experiments"]).__repr__()
                                          for o in objects.values())),
            "scorer_reproducibility": reproducibility,
            "easy_candidate_pool": {"objects": len(objects), "samples": len(manifest)}}
