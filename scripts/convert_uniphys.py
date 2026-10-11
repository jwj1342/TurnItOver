#!/usr/bin/env python3
"""Convert selected UniPhys URDF assets into a frozen split-aware catalog."""
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


def _load_split_plan(path: Path) -> tuple[tuple[str, ...], dict[str, str], dict]:
    record = json.loads(path.read_text(encoding="utf-8"))
    rows = record.get("objects")
    if record.get("version") != 1 or not isinstance(rows, list) or not rows:
        raise ValueError("Split plan requires a nonempty version-1 object list")
    assignments = {}
    for row in rows:
        object_id, split = row.get("id"), row.get("split")
        if (not isinstance(object_id, str) or parse_ids(object_id) != [object_id]
                or object_id in assignments or split not in {"train", "validation"}):
            raise ValueError("Split plan contains duplicate IDs or unsupported splits")
        assignments[object_id] = split
    test_ids = record.get("test_object_ids")
    if (not isinstance(test_ids, list) or not test_ids
            or any(not isinstance(value, str) for value in test_ids)
            or set(assignments) & set(test_ids)):
        raise ValueError("Split plan must identify a disjoint frozen test object set")
    expected = record.get("expected_object_counts")
    actual = {split: sum(value == split for value in assignments.values())
              for split in ("train", "validation")}
    if expected != actual:
        raise ValueError(f"Split plan counts disagree: expected {expected}, got {actual}")
    return tuple(assignments), assignments, record


def convert(root: Path, output: Path, object_ids: tuple[str, ...] = DEFAULT_IDS, *,
            split_assignments: dict[str, str] | None = None, split_plan: dict | None = None) -> Path:
    acquisition_path = root / "download-manifest.json"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    revision = acquisition.get("revision")
    if (acquisition.get("dataset") != "breezexian/UniPhys-Bench-Part2" or
            acquisition.get("license") != "CC-BY-NC-4.0" or
            not isinstance(revision, str) or len(revision) != 40):
        raise ValueError("Expected a pinned UniPhys-Bench Part 2 acquisition")
    if output.exists():
        raise ValueError("Output must not exist")
    assignments = (dict(split_assignments) if split_assignments is not None
                   else {object_id: "test" for object_id in object_ids})
    if (set(assignments) != set(object_ids) or len(assignments) != len(object_ids)
            or set(assignments.values()) not in ({"test"}, {"train", "validation"})):
        raise ValueError("Every selected object must map exactly once to test or train/validation")
    evaluation_only = set(assignments.values()) == {"test"}
    if split_plan is not None and split_plan.get("dataset_revision") != revision:
        raise ValueError("Split plan dataset revision differs from the acquisition")
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
            "evaluation_only": evaluation_only,
            "role": "evaluation_only" if evaluation_only else "train_validation",
            "assets": entries,
        }
        (output / "catalog.json").write_text(
            json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        split_name = "test-splits.json" if evaluation_only else "trainval-splits.json"
        split_record = {
            "version": 1,
            "unit": "split_group",
            "groups": {entry["split_group"]: assignments[object_id]
                       for entry, object_id in zip(entries, object_ids)},
        }
        if not evaluation_only:
            split_record["expected_object_counts"] = {
                split: sum(value == split for value in assignments.values())
                for split in ("train", "validation")
            }
        (output / split_name).write_text(json.dumps(split_record, indent=2) + "\n", encoding="utf-8")
        (output / "conversion.json").write_text(json.dumps({
            "version": 1,
            "status": "complete",
            "dataset": acquisition["dataset"],
            "dataset_revision": revision,
            "license": acquisition["license"],
            "role": "evaluation_only" if evaluation_only else "train_validation",
            "split_plan": split_plan,
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
    parser.add_argument("--ids", help="comma-separated IDs; defaults to the frozen test IDs")
    parser.add_argument("--split-plan", type=Path,
                        help="frozen train/validation selection JSON; cannot be combined with --ids")
    args = parser.parse_args()
    if args.ids and args.split_plan:
        parser.error("--ids and --split-plan are mutually exclusive")
    if args.split_plan:
        object_ids, assignments, plan = _load_split_plan(args.split_plan.resolve())
    else:
        object_ids, assignments, plan = tuple(parse_ids(args.ids or ",".join(DEFAULT_IDS))), None, None
    print(convert(args.root.resolve(), args.out.resolve(), object_ids,
                  split_assignments=assignments, split_plan=plan))


if __name__ == "__main__":
    main()
