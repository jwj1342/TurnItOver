"""Freeze new official sources that can host v1.1 composite Hard faults."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from medium_expansion_sources import load_exclusions, sha
from turnitover.assets.catalog import decode_spec


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def legal_targets(spec: object) -> list[dict]:
    targets = []
    for joint in spec.joints:
        lo, hi = map(float, joint.limits)
        endpoint = max((lo, hi), key=abs)
        if not lo <= 0 <= hi or abs(endpoint) <= 1e-8:
            continue
        if joint.type.value == "revolute" and abs(endpoint) > 3.141592653589793:
            continue
        if joint.type.value not in ("revolute", "prismatic"):
            continue
        targets.append({"joint_id": joint.id, "part_id": joint.part,
                        "joint_type": joint.type.value, "original_limits": [lo, hi]})
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--conversion", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-joints", type=int, default=16)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be new")
    excluded, pins, blocked_geometry = load_exclusions(args.exclude)
    catalog = {Path(row["spec"]).stem: row for row in json.loads(args.catalog.read_text())["assets"]}
    conversion = json.loads(args.conversion.read_text())["assets"]
    manifest = json.loads(args.download_manifest.read_text())
    archives = {str(row["asset_id"]): row for row in manifest["archives"]}
    pins.update({str(path.resolve()): sha(path) for path in
                 (args.catalog, args.conversion, args.download_manifest, Path(__file__))})
    root, pool, rejected = args.download_manifest.resolve().parent, [], []
    for record in conversion:
        asset_id, identity = str(record.get("asset_id", "")), record.get("source_identity")
        pm_id = "PM_" + asset_id
        geometry = record.get("scale_audit", {}).get("geometry_fingerprint")
        if (record.get("status") != "structural_candidate" or record.get("frozen_test_overlap") is not False
                or identity in excluded or geometry in blocked_geometry or pm_id not in catalog
                or not 1 <= record.get("joints", 0) <= args.max_joints):
            continue
        spec_path, urdf = args.catalog.resolve().parent / catalog[pm_id]["spec"], root / "extracted" / asset_id / "mobility.urdf"
        if not urdf.is_file() or sha(spec_path) != record["spec_sha256"]:
            rejected.append({"asset_id": pm_id, "reason": "source_binding_changed"}); continue
        spec, targets = decode_spec(json.loads(spec_path.read_text())), None
        targets = legal_targets(spec)
        if not targets:
            rejected.append({"asset_id": pm_id, "reason": "no_legal_motion_target"}); continue
        archive = root / archives[asset_id]["path"]
        metadata = {"dataset": manifest["dataset"], "source_dataset": manifest["dataset"],
                    "dataset_revision": manifest["dataset_revision"], "license": manifest["license"],
                    "source_identity": identity, "source_asset_id": asset_id, "model_id": record["model_id"],
                    "archive_sha256": record["archive_sha256"], "original_spec_sha256": record["spec_sha256"],
                    "original_geometry_fingerprint": geometry, "source_spec": str(spec_path),
                    "source_archive": str(archive)}
        source_record = args.out / "qualified" / f"{pm_id}.json"
        files = {str(path): sha(path) for path in urdf.parent.rglob("*") if path.is_file()}
        files[str(spec_path)] = record["spec_sha256"]
        write(source_record, {"asset_id": pm_id, "metadata": metadata, "files": files,
                              "conversion_record": record})
        pool.append({"asset_id": pm_id, "source_identity": identity, "category": spec.category,
                     "active_joints": len(spec.joints), "parts": len(spec.parts),
                     "source_spec": str(spec_path), "source_urdf": str(urdf),
                     "source_record": str(source_record), "available_targets": targets,
                     "qualification": "v1.1 compositional source only; render and human review pending"})
    pool.sort(key=lambda row: hashlib.sha256(row["source_identity"].encode()).hexdigest())
    write(args.out / "pool.json", pool); write(args.out / "rejections.json", rejected)
    write(args.out / "manifest.json", {"version": 1, "policy": "hard-full-motion-formal-v1.1",
          "inputs": pins, "accepted": len(pool), "rejected": len(rejected),
          "files": {p.relative_to(args.out).as_posix(): sha(p) for p in args.out.rglob("*") if p.is_file()}})
    print(json.dumps({"accepted": len(pool), "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
