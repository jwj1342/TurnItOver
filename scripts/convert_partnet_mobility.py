#!/usr/bin/env python3
"""Convert and qualify official PartNet-Mobility ZIPs for formal Train/Validation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
import trimesh

from turnitover.assets.urdf import UrdfModel
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.serde import to_dict

DATASET = "sapien-sim/PartNetMobility"
LICENSE = "PartNet-Mobility terms: non-commercial research/education; ShapeNet terms also apply"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, root: Path, asset_id: str) -> Path:
    destination = root / asset_id
    if destination.is_dir():
        return destination
    temporary = root / f".{asset_id}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                parts = Path(item.filename).parts
                if (not parts or parts[0] != asset_id or ".." in parts
                        or Path(item.filename).is_absolute()
                        or (item.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError("ZIP contains an unsafe or unexpected path")
            zf.extractall(temporary)
        extracted = temporary / asset_id
        if not extracted.is_dir():
            raise ValueError("ZIP does not contain its declared asset directory")
        extracted.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def _geometry_fingerprint(vertices: np.ndarray) -> str:
    quantized = np.unique(np.round(np.asarray(vertices, dtype=np.float64), 6), axis=0)
    order = np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))
    return hashlib.sha256(quantized[order].tobytes()).hexdigest()


def _test_identities(plan: Path, root: Path) -> dict[str, set[str]]:
    record = json.loads(plan.read_text(encoding="utf-8"))
    if record.get("role") != "formal_test":
        raise ValueError("Expected frozen formal Test plan")
    result = {"source_ids": set(), "geometry": set()}
    for row in record["objects"]:
        folder = root / row["id"]
        metadata = json.loads((folder / "meta_data.json").read_text(encoding="utf-8"))
        source = metadata["source"]
        result["source_ids"].add(f"{source['dataset']}:{source['original_id']}")
        scene = trimesh.load(folder / "full_model/model.obj", force="scene", process=False)
        vertices = np.concatenate([mesh.vertices for mesh in scene.geometry.values()])
        result["geometry"].add(_geometry_fingerprint(vertices))
    return result


def _scale_audit(folder: Path, model: UrdfModel) -> dict:
    declared = json.loads((folder / "bounding_box.json").read_text(encoding="utf-8"))
    expected = np.asarray(declared["max"], dtype=float) - np.asarray(declared["min"], dtype=float)
    world = model.world_vertices({})
    vertices = np.concatenate(list(world.values()))
    observed = vertices.max(axis=0) - vertices.min(axis=0)
    relative = np.abs(np.sort(observed) - np.sort(expected)) / np.maximum(np.sort(expected), 1e-9)
    if not np.isfinite(relative).all() or relative.max() > 0.02:
        raise ValueError("URDF rest geometry disagrees with official bounding_box.json by more than 2%")
    largest = float(observed.max())
    if not 0.02 <= largest <= 5.0:
        raise ValueError("implausible object scale in metres")
    return {"declared_extents": expected.tolist(), "observed_extents": observed.tolist(),
            "sorted_extent_max_relative_error": float(relative.max()),
            "largest_extent_m": largest, "geometry_fingerprint": _geometry_fingerprint(vertices)}


def _annotation_audit(folder: Path, spec) -> dict:
    rows = json.loads((folder / "mobility_v2.json").read_text(encoding="utf-8"))
    expected = Counter({"hinge": "revolute", "slider": "prismatic"}[row["joint"]]
                       for row in rows if row.get("joint") in ("hinge", "slider"))
    observed = Counter(joint.type.value for joint in spec.joints)
    if expected != observed:
        raise ValueError(f"URDF joint types disagree with mobility_v2.json: {observed} != {expected}")
    return {"mobility_v2_joint_types": dict(expected), "urdf_joint_types": dict(observed),
            "agreement": True}


def _reviews(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("version") != 1 or not isinstance(record.get("assets"), list):
        raise ValueError("Review manifest requires version=1 and assets")
    result = {}
    for row in record["assets"]:
        asset_id = row.get("asset_id")
        if (not re.fullmatch(r"PM_\d+", str(asset_id)) or asset_id in result
                or row.get("decision") not in ("passed", "failed")
                or not str(row.get("reason", "")).strip()):
            raise ValueError("Invalid or duplicate motion review")
        if row["decision"] == "passed" and row.get("split") not in ("train", "validation"):
            raise ValueError("Passed review requires a Train/Validation split")
        if row["decision"] == "failed":
            result[asset_id] = dict(row)
            continue
        smoke = Path(row["smoke_json"])
        if not smoke.is_absolute():
            smoke = Path(__file__).resolve().parents[1] / smoke
        if sha256(smoke) != row.get("smoke_sha256"):
            raise ValueError(f"Motion-review hash mismatch: {asset_id}")
        payload = json.loads(smoke.read_text(encoding="utf-8"))
        if payload.get("asset_id") != asset_id or payload.get("status") != "mechanical_checks_complete":
            raise ValueError(f"Incompatible smoke evidence: {asset_id}")
        result[asset_id] = {**row, "smoke_json": str(smoke.resolve())}
    return result


def convert(root: Path, output: Path, test_plan: Path, test_root: Path,
            *, reviews_path: Path | None = None, max_joints: int = 8,
            excluded_source_ids: set[str] | None = None) -> Path:
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    manifest = json.loads((root / "download-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("dataset") != DATASET:
        raise ValueError("Complete official PartNet-Mobility download manifest required")
    reviews = _reviews(reviews_path)
    test = _test_identities(test_plan, test_root)
    excluded_source_ids = excluded_source_ids or set()
    output.mkdir(parents=True)
    extract_root = root / "extracted"
    extract_root.mkdir(exist_ok=True)
    reports, candidate_entries, entries, structural_ids = [], [], [], set()
    archive_rows = {row["asset_id"]: row for row in manifest["archives"]}
    for asset_id in sorted(archive_rows, key=int):
        row = archive_rows[asset_id]
        archive = root / row["path"]
        report = {"asset_id": asset_id, "source_identity": f"PartNet-Mobility:{asset_id}"}
        try:
            if sha256(archive) != row["sha256"]:
                raise ValueError("archive SHA-256 differs from download manifest")
            if asset_id in excluded_source_ids:
                raise ValueError("source identity overlaps an existing formal Train/Validation asset")
            folder = _safe_extract(archive, extract_root, asset_id)
            metadata = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            if (not str(metadata.get("anno_id", "")).strip()
                    or not str(metadata.get("model_id", "")).strip()
                    or not metadata.get("model_cat")):
                raise ValueError("meta.json provenance/category is incomplete")
            model = UrdfModel(folder / "mobility.urdf")
            spec = model.spec(f"PM_{asset_id}", metadata["model_cat"])
            if not 1 <= len(spec.joints) <= max_joints:
                raise ValueError(f"active joint count outside 1..{max_joints}")
            annotation = _annotation_audit(folder, spec)
            scale = _scale_audit(folder, model)
            if report["source_identity"] in test["source_ids"] or scale["geometry_fingerprint"] in test["geometry"]:
                raise ValueError("asset overlaps frozen Part 2 Test by source or geometry")
            structural_ids.add(spec.asset_id)
            spec_name = f"{spec.asset_id}.json"
            write_json_atomic(output / spec_name, to_dict(spec))
            review = reviews.get(spec.asset_id)
            eligible = review is not None and review["decision"] == "passed"
            report.update(status="structural_candidate", category=spec.category,
                          annotation_id=str(metadata["anno_id"]), model_id=metadata["model_id"],
                          archive_sha256=row["sha256"],
                          spec_sha256=sha256(output / spec_name), parts=len(spec.parts),
                          joints=len(spec.joints), joint_types=dict(Counter(j.type.value for j in spec.joints)),
                          triangles=sum(len(part.mesh.faces) for part in spec.parts),
                          scale_audit=scale, annotation_audit=annotation,
                          frozen_test_overlap=False, motion_review=review,
                          formal_train_validation_eligible=eligible)
            base_entry = {
                "spec": spec_name, "split_group": report["source_identity"],
                "origin": f"{DATASET}@{manifest['dataset_revision']} {asset_id}.zip",
                "license": LICENSE, "formal_training": eligible,
                "dataset": DATASET, "dataset_revision": manifest["dataset_revision"],
                "source_identity": report["source_identity"],
                "annotation_id": report["annotation_id"], "model_id": report["model_id"],
                "archive_sha256": row["sha256"], "spec_sha256": report["spec_sha256"],
            }
            candidate_entries.append(base_entry)
            if eligible:
                entries.append(base_entry)
        except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
            report.update(status="rejected", rejection_reason=str(exc),
                          formal_train_validation_eligible=False)
        reports.append(report)
    invalid_passes = {key for key, row in reviews.items() if row["decision"] == "passed"} - structural_ids
    if invalid_passes:
        raise ValueError(f"Reviews pass non-structural assets: {sorted(invalid_passes)}")
    role = "formal_train_validation" if reviews_path is not None else "structural_candidates"
    write_json_atomic(output / "catalog.json", {
        "version": 1, "source_id": f"partnet-mobility-{manifest['dataset_revision'][:12]}",
        "role": role, "formal_training": reviews_path is not None,
        "assets": entries if reviews_path is not None else candidate_entries,
    })
    if reviews_path is not None:
        write_json_atomic(output / "splits.json", {
            "version": 1, "role": role, "unit": "split_group",
            "groups": {entry["split_group"]: reviews[Path(entry["spec"]).stem]["split"]
                       for entry in entries},
        })
    reasons = Counter(row.get("rejection_reason", "structural_candidate") for row in reports)
    write_json_atomic(output / "conversion.json", {
        "version": 1, "status": "complete", "dataset": DATASET,
        "dataset_revision": manifest["dataset_revision"], "role": role,
        "formal_training": reviews_path is not None, "license": LICENSE,
        "formal_test_plan_sha256": sha256(test_plan), "assets": reports,
        "summary": {"selected": len(reports), "structural_candidates": len(structural_ids),
                    "training_eligible": len(entries), "reasons": dict(reasons)},
    })
    return output / "catalog.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/partnet-mobility-pilot"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, default=Path("configs/uniphys-test.json"))
    parser.add_argument("--test-root", type=Path, default=Path("data/assets/uniphys/data"))
    parser.add_argument("--motion-reviews", type=Path)
    parser.add_argument("--max-joints", type=int, default=8)
    parser.add_argument("--exclude-source-ids", default="6635,6659")
    args = parser.parse_args()
    excluded = {value.strip() for value in args.exclude_source_ids.split(",") if value.strip()}
    print(convert(args.root.resolve(), args.out.resolve(), args.test_plan.resolve(),
                  args.test_root.resolve(),
                  reviews_path=args.motion_reviews.resolve() if args.motion_reviews else None,
                  max_joints=args.max_joints, excluded_source_ids=excluded))


if __name__ == "__main__":
    main()
