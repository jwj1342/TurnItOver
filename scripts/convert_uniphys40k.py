#!/usr/bin/env python3
"""Batch-convert only evidence-qualified UniPhys-40K ZIPs into a catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.assets.uniphys40k import UnsupportedAsset, convert_archive
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.serde import to_dict


DATASET = "breezexian/UniPhys-40K"
KINEMATIC_SEMANTICS = {
    "evidence": "breezexian/UniPhysGen@3f1bb2dd4c2e79b4c21c2617de99d8d0160150b7",
    "source_obj_pose": "qpos_zero",
    "axis_and_pivot_frame": "released_obj_coordinates",
    "prismatic_pivot": "not_applicable; official MuJoCo export uses joint pos 0 0 0",
    "revolute_range_unit": "radian",
    "prismatic_range_unit": "released_obj_coordinate_unit",
    "mapping": "source_qpos_relative_to_released_obj_zero_pose",
}
LICENSE_BY_SOURCE = {
    "ABO": {
        "catalog": "UniPhys annotations: CC-BY-4.0; mesh: ABO CC-BY-4.0",
        "urls": [
            "https://huggingface.co/datasets/breezexian/UniPhys-40K/blob/main/LICENSE",
            "https://amazon-berkeley-objects.s3.amazonaws.com/index.html",
        ],
    },
    "3D-FUTURE": {
        "catalog": (
            "UniPhys annotations: CC-BY-4.0; mesh: 3D-FUTURE Data Sets Use License "
            "Agreement (scientific research only; acceptance required; no commercial or external-service use)"
        ),
        "urls": [
            "https://huggingface.co/datasets/breezexian/UniPhys-40K/blob/main/LICENSE",
            "https://github.com/3D-FRONT-FUTURE/3D-FUTURE-ToolBox",
            "https://terms.aliyun.com/legal-agreement/terms/suit_bu1_ali_cloud/suit_bu1_ali_cloud202004171628_60052.html",
        ],
    },
    "HSSD": {
        "catalog": "UniPhys annotations: CC-BY-4.0; mesh: HSSD CC-BY-NC-4.0",
        "urls": [
            "https://huggingface.co/datasets/breezexian/UniPhys-40K/blob/main/LICENSE",
            "https://huggingface.co/datasets/hssd/hssd-hab/blob/main/README.md",
        ],
    },
    "PartNet": {
        "catalog": "UniPhys annotations: CC-BY-4.0; mesh: PartNet/ShapeNet non-commercial terms",
        "urls": [
            "https://huggingface.co/datasets/breezexian/UniPhys-40K/blob/main/LICENSE",
            "https://huggingface.co/datasets/ShapeNet/PartNet-archive/blob/main/README.md",
            "https://shapenet.org/terms",
        ],
    },
}


def _revision(root: Path, archive: Path) -> str:
    relative = archive.resolve().relative_to(root.resolve())
    metadata = root / ".cache/huggingface/download" / Path(str(relative) + ".metadata")
    try:
        revision = metadata.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError) as exc:
        raise UnsupportedAsset("missing Hugging Face revision metadata") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise UnsupportedAsset("invalid Hugging Face revision metadata")
    return revision


def _test_identities(plan: Path, root: Path) -> dict[str, set[str]]:
    record = json.loads(plan.read_text(encoding="utf-8"))
    if record.get("role") != "formal_test" or not isinstance(record.get("objects"), list):
        raise ValueError("Expected the frozen formal Test plan")
    result = {"source": set(), "name": set(), "geometry": set()}
    for row in record["objects"]:
        folder = root / row["id"]
        metadata = json.loads((folder / "meta_data.json").read_text(encoding="utf-8"))
        source = metadata["source"]
        result["source"].add(f"{source['dataset']}:{source['original_id']}")
        result["name"].add(re.sub(r"[^a-z0-9]+", "", metadata["object"]["object_name"].lower()))
        result["geometry"].add(hashlib.sha256((folder / "full_model/model.obj").read_bytes()).hexdigest())
    return result


def _archive_identity(archive: Path) -> dict:
    import zipfile
    with zipfile.ZipFile(archive) as zf:
        names = [name for name in zf.namelist() if name.endswith("/meta_data.json")]
        if len(names) != 1:
            raise UnsupportedAsset("expected one meta_data.json")
        metadata = json.loads(zf.read(names[0]))
    source = metadata.get("source") or {}
    if not all(isinstance(source.get(key), str) and source[key].strip()
               for key in ("dataset", "original_id", "license_ref")):
        raise UnsupportedAsset("source provenance is incomplete")
    return {"source": source, "object_name": (metadata.get("object") or {}).get("object_name")}


def _overlap(identity: dict, identities: dict[str, set[str]], archive: Path) -> dict:
    source = identity["source"]
    values = {
        "source": f"{source['dataset']}:{source['original_id']}",
        "name": re.sub(r"[^a-z0-9]+", "", str(identity["object_name"]).lower()),
        "geometry": _full_model_sha(archive),
    }
    matches = {key: value in identities[key] for key, value in values.items()}
    return {"policy": ["source_identity", "full_model_sha256", "normalized_object_name"],
            "values": values, "matches_formal_test": matches, "overlap": any(matches.values())}


def _full_model_sha(path: Path) -> str:
    import zipfile
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if name.endswith("/full_model/model.obj")]
        if len(names) != 1:
            raise UnsupportedAsset("expected one full_model/model.obj")
        return hashlib.sha256(zf.read(names[0])).hexdigest()


def _motion_reviews(path: Path) -> dict[str, dict]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("version") != 1 or not isinstance(record.get("assets"), list):
        raise ValueError("Motion review manifest requires version=1 and assets")
    result = {}
    for row in record["assets"]:
        asset_id = row.get("asset_id")
        if (not isinstance(asset_id, str) or asset_id in result
                or row.get("decision") not in ("passed", "failed")
                or not isinstance(row.get("reason"), str) or not row["reason"].strip()):
            raise ValueError("Invalid or duplicate motion review")
        smoke_path = Path(row.get("smoke_json", ""))
        if not smoke_path.is_absolute():
            smoke_path = REPO_ROOT / smoke_path
        smoke_bytes = smoke_path.read_bytes()
        if hashlib.sha256(smoke_bytes).hexdigest() != row.get("smoke_sha256"):
            raise ValueError(f"Motion review smoke hash mismatch: {asset_id}")
        smoke = json.loads(smoke_bytes)
        if (smoke.get("asset_id") != asset_id or smoke.get("status") != "mechanical_checks_complete"
                or smoke.get("physical_motion_review") != "required"):
            raise ValueError(f"Motion review references incompatible smoke evidence: {asset_id}")
        if row["decision"] == "passed" and row.get("split") not in ("train", "validation"):
            raise ValueError(f"Passed motion review requires train/validation split: {asset_id}")
        result[asset_id] = {**row, "smoke_json": str(smoke_path.resolve())}
    return result


def convert(root: Path, output: Path, test_plan: Path, test_root: Path,
            selected_ids: set[str] | None = None, reviews_path: Path | None = None) -> Path:
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    archives = sorted(root.rglob("UP_*.zip"))
    if selected_ids is not None:
        archives = [path for path in archives if path.stem in selected_ids]
        missing = selected_ids - {path.stem for path in archives}
        if missing:
            raise ValueError(f"Selected ZIPs not found: {sorted(missing)}")
    if not archives:
        raise ValueError("No UniPhys-40K ZIPs selected")
    reviews = _motion_reviews(reviews_path) if reviews_path is not None else None
    identities = _test_identities(test_plan, test_root)
    output.mkdir(parents=True)
    entries, rows, revisions, structural_ids = [], [], set(), set()
    for archive in archives:
        audit = {}
        try:
            revision = _revision(root, archive)
            revisions.add(revision)
            identity = _archive_identity(archive)
            source = identity["source"]
            license_record = LICENSE_BY_SOURCE.get(source["license_ref"])
            if license_record is None or source["dataset"] != source["license_ref"]:
                raise UnsupportedAsset(f"unsupported upstream license_ref: {source['license_ref']}")
            overlap = _overlap(identity, identities, archive)
            audit = {**identity, "dataset_revision": revision, "license": license_record,
                     "formal_test_overlap": overlap}
            if overlap["overlap"]:
                raise UnsupportedAsset("asset overlaps the frozen formal Test by source, geometry, or name")
            converted = convert_archive(archive)
            structural_ids.add(converted.spec.asset_id)
            review = reviews.get(converted.spec.asset_id) if reviews is not None else None
            eligible = review is not None and review["decision"] == "passed"
            qualification = dict(converted.report["qualification"])
            if reviews is not None:
                qualification.update(rendered_motion=(review or {}).get("decision", "not_reviewed"),
                                     formal_train_validation_eligible=eligible)
            report = {**converted.report, "qualification": qualification, **audit}
            if review is not None:
                report["motion_review"] = review
            rows.append(report)
            if reviews is not None and not eligible:
                continue
            spec_name = f"{converted.spec.asset_id}.json"
            write_json_atomic(output / spec_name, to_dict(converted.spec))
            source_key = f"{source['dataset']}:{source['original_id']}"
            entries.append({
                "spec": spec_name,
                "split_group": f"uniphys40k:{source_key}",
                "origin": f"{DATASET}@{revision}:{converted.spec.asset_id}; upstream {source_key}",
                "license": license_record["catalog"],
            })
        except (OSError, TypeError, ValueError, KeyError) as exc:
            rows.append({"asset_id": archive.stem, "archive": str(archive.resolve()),
                         "status": "unsupported", "reason": str(exc),
                         "error_type": type(exc).__name__, **audit,
                         "qualification": {"rendered_motion": "not_run",
                                           "formal_train_validation_eligible": False}})
    if len(revisions) > 1:
        raise ValueError(f"Selected ZIPs mix dataset revisions: {sorted(revisions)}")
    invalid_passes = ({key for key, value in (reviews or {}).items()
                       if value["decision"] == "passed"} - structural_ids)
    if invalid_passes:
        raise ValueError(f"Motion review passed non-structural assets: {sorted(invalid_passes)}")
    if not entries:
        write_json_atomic(output / "conversion.json", {
            "version": 1, "status": "failed", "dataset": DATASET,
            "kinematic_semantics": KINEMATIC_SEMANTICS, "assets": rows,
        })
        raise ValueError("No selected archive passed conversion")
    revision = next(iter(revisions))
    role = "formal_train_validation" if reviews is not None else "diagnostic_conversion_candidates"
    write_json_atomic(output / "catalog.json", {
        "version": 1, "source_id": f"uniphys40k-{revision[:12]}",
        "role": role, "assets": entries,
    })
    if reviews is not None:
        groups = {entry["split_group"]: reviews[Path(entry["spec"]).stem]["split"]
                  for entry in entries}
        write_json_atomic(output / "splits.json", {
            "version": 1, "role": role, "unit": "split_group", "groups": groups,
        })
    write_json_atomic(output / "conversion.json", {
        "version": 1, "status": "complete", "dataset": DATASET,
        "dataset_revision": revision, "role": role,
        "kinematic_semantics": KINEMATIC_SEMANTICS,
        "formal_test_plan": str(test_plan.resolve()), "assets": rows,
        "summary": {"selected": len(archives), "structural_candidates": len(structural_ids),
                    "training_eligible": len(entries) if reviews is not None else 0,
                    "unsupported": len(archives) - len(structural_ids)},
    })
    return output / "catalog.json"


def _ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    if not result or any(not re.fullmatch(r"UP_\d{8}", item) for item in result):
        raise ValueError("--ids requires comma-separated UP_######## IDs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/uniphys40k-pilot"))
    parser.add_argument("--out", type=Path, default=Path("data/assets/uniphys40k-pilot-catalog"))
    parser.add_argument("--ids", help="comma-separated archive IDs; default: all ZIPs below --root")
    parser.add_argument("--test-plan", type=Path, default=Path("configs/uniphys-test.json"))
    parser.add_argument("--test-root", type=Path, default=Path("data/assets/uniphys/data"))
    parser.add_argument("--motion-reviews", type=Path,
                        help="review manifest required to emit a formal Train/Validation catalog")
    args = parser.parse_args()
    result = convert(args.root.resolve(), args.out.resolve(), args.test_plan.resolve(),
                     args.test_root.resolve(), _ids(args.ids),
                     args.motion_reviews.resolve() if args.motion_reviews else None)
    print(result)


if __name__ == "__main__":
    main()
