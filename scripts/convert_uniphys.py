#!/usr/bin/env python3
"""Convert selected UniPhys URDF assets into a frozen evaluation catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.download_uniphys import parse_ids
from turnitover.assets.urdf import UrdfModel
from turnitover.core.serde import to_dict


DEFAULT_IDS = (
    "UPB_00000000", "UPB_00000002", "UPB_00000004", "UPB_00000006",
    "UPB_00000007", "UPB_00000009", "UPB_00000010", "UPB_00000012",
    "UPB_00000014", "UPB_00000015", "UPB_00000016", "UPB_00000017",
    "UPB_00000018", "UPB_00000019", "UPB_00000020", "UPB_00000021",
    "UPB_00000022", "UPB_00000025", "UPB_00000026", "UPB_00000027",
)


def _digest(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def convert(root: Path, output: Path, object_ids: tuple[str, ...] = DEFAULT_IDS) -> Path:
    acquisition_path = root / "download-manifest.json"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    revision = acquisition.get("revision")
    if (acquisition.get("dataset") != "breezexian/UniPhys-Bench-Part2" or
            acquisition.get("license") != "CC-BY-NC-4.0" or
            not isinstance(revision, str) or len(revision) != 40):
        raise ValueError("Expected a pinned UniPhys-Bench Part 2 acquisition")
    if output.exists():
        raise ValueError("Output must not exist")
    output.mkdir(parents=True)

    downloaded = {row["object_id"]: row for row in acquisition.get("results", [])}
    entries, reports = [], []
    try:
        for object_id in object_ids:
            folder = root / "data" / object_id
            metadata_path = folder / "meta_data.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model = UrdfModel(folder / "model.urdf", continuous_joint_policy="fixed")
            spec = model.spec(object_id, metadata["object"]["category"])
            spec_path = f"{object_id}.json"
            (output / spec_path).write_text(
                json.dumps(to_dict(spec), separators=(",", ":")), encoding="utf-8")

            inputs = model.provenance() + [{
                "path": str(metadata_path.resolve()),
                "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            }]
            inputs = sorted(({
                "path": Path(row["path"]).resolve().relative_to(root.resolve()).as_posix(),
                "sha256": row["sha256"],
            } for row in inputs), key=lambda row: row["path"])
            asset_sha = _digest(inputs)
            source_record = downloaded.get(object_id)
            recorded = ({row["path"]: row["sha256"] for row in source_record.get("records", [])}
                        if source_record and source_record.get("status") == "complete" else {})
            manifest_verified = bool(recorded) and all(recorded.get(row["path"]) == row["sha256"]
                                                       for row in inputs)
            entries.append({
                "spec": spec_path,
                "split_group": f"uniphys:{revision}:{object_id}",
                "origin": f"{acquisition['dataset']}@{revision} data/{object_id}",
                "license": acquisition["license"],
            })
            reports.append({
                "asset_id": object_id,
                "asset_sha256": asset_sha,
                "inputs": inputs,
                "download_manifest_verified": manifest_verified,
                "continuous_joint_policy": "fixed_at_zero",
                "continuous_joints_fixed_at_zero": list(model.continuous_joints_fixed_at_zero),
                "parts": len(spec.parts),
                "joints": len(spec.joints),
            })

        catalog = {
            "version": 1,
            "source_id": f"uniphys-{revision}",
            "dataset_revision": revision,
            "evaluation_only": True,
            "assets": entries,
        }
        (output / "catalog.json").write_text(
            json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        (output / "test-splits.json").write_text(json.dumps({
            "version": 1,
            "unit": "split_group",
            "groups": {entry["split_group"]: "test" for entry in entries},
        }, indent=2) + "\n", encoding="utf-8")
        (output / "conversion.json").write_text(json.dumps({
            "version": 1,
            "status": "complete",
            "dataset": acquisition["dataset"],
            "dataset_revision": revision,
            "license": acquisition["license"],
            "role": "evaluation_only",
            "continuous_joint_policy": "fixed_at_zero",
            "assets": reports,
        }, indent=2) + "\n", encoding="utf-8")
        return output / "catalog.json"
    except Exception:
        (output / "conversion.json").write_text(
            json.dumps({"version": 1, "status": "incomplete"}, indent=2) + "\n",
            encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/assets/uniphys"))
    parser.add_argument("--out", type=Path, default=Path("data/assets/uniphys-catalog-v1"))
    parser.add_argument("--ids", default=",".join(DEFAULT_IDS))
    args = parser.parse_args()
    print(convert(args.root.resolve(), args.out.resolve(), tuple(parse_ids(args.ids))))


if __name__ == "__main__":
    main()
