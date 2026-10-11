#!/usr/bin/env python3
"""Build frozen UniPhys clean/axis/offset verifier cases from a split-aware catalog."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.assets.catalog import CatalogSource
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.corruptions import get_corruption
from turnitover.render.session import ObservationSession, RenderConfig
from turnitover.render.views import load_views
from turnitover.telemetry import git_state, now_iso
from turnitover.verifier.contracts import VERIFIER_SCHEMA_VERSION
from turnitover.verifier.gt_validator import validate_code


OBJECT_COUNT = 20
MAX_TRIANGLES = 1_000_000
MAX_DRAW_CALLS = 1_024


def _validate_case_identity(cases: list[dict], gold: dict[str, list[dict]]) -> None:
    gold_rows = [row for rows in gold.values() for row in rows]
    case_ids = [row["id"] for row in cases]
    gold_by_id = {row["id"]: row for row in gold_rows}
    if len(set(case_ids)) != len(case_ids) or len(gold_by_id) != len(gold_rows):
        raise ValueError("Benchmark case IDs must be unique")
    if set(case_ids) != set(gold_by_id):
        raise ValueError("Benchmark and gold case IDs differ")
    program_shas = [row["program_sha"] for row in gold_rows]
    if len(set(program_shas)) != len(program_shas):
        raise ValueError("Benchmark candidate program SHAs must be unique")
    variants_by_group = defaultdict(list)
    for case in cases:
        target = gold_by_id[case["id"]]
        if (case["split"] != target["split"]
                or case["split_group"] != target["split_group"]):
            raise ValueError(f"Case split identity mismatch: {case['id']}")
        variants_by_group[case["split_group"]].append(
            (case["id"].rsplit("-", 1)[-1], case["split"]))
    if any(sorted(variant for variant, _ in rows) != ["axis", "clean", "offset"]
           or len({split for _, split in rows}) != 1
           for rows in variants_by_group.values()):
        raise ValueError("Each split_group must contain clean/axis/offset exactly once")


def _load_frozen_catalog(catalog_path: Path) -> tuple[dict, CatalogSource, tuple]:
    record = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not record.get("evaluation_only") or not record.get("dataset_revision"):
        raise ValueError("Expected a pinned evaluation-only UniPhys catalog")
    source = CatalogSource(str(catalog_path))
    specs = tuple(source)
    split_groups = [source.metadata(spec.asset_id)["split_group"] for spec in specs]
    if len(specs) != OBJECT_COUNT:
        raise ValueError(f"Expected {OBJECT_COUNT} catalog assets, got {len(specs)}")
    if len(set(split_groups)) != len(split_groups):
        raise ValueError("Each catalog asset must have a unique split_group")
    return record, source, specs


def _load_catalog(catalog_path: Path, splits_path: Path | None):
    if splits_path is None:
        record, source, specs = _load_frozen_catalog(catalog_path)
        return record, source, specs, {
            source.metadata(spec.asset_id)["split_group"]: "test" for spec in specs}
    record = json.loads(catalog_path.read_text(encoding="utf-8"))
    if (record.get("evaluation_only") is not False or record.get("role") != "train_validation"
            or not record.get("dataset_revision")):
        raise ValueError("Expected a pinned train/validation UniPhys catalog")
    source = CatalogSource(str(catalog_path))
    specs = tuple(source)
    split_record = json.loads(splits_path.read_text(encoding="utf-8"))
    groups = split_record.get("groups")
    if (split_record.get("version") != 1 or split_record.get("unit") != "split_group"
            or not isinstance(groups, dict) or set(groups.values()) != {"train", "validation"}):
        raise ValueError("Expected a train/validation split-group manifest")
    catalog_groups = {source.metadata(spec.asset_id)["split_group"] for spec in specs}
    if set(groups) != catalog_groups or len(catalog_groups) != len(specs):
        raise ValueError("Catalog and split manifest groups differ or are not unique")
    actual = {split: sum(value == split for value in groups.values())
              for split in ("train", "validation")}
    if split_record.get("expected_object_counts") != actual:
        raise ValueError("Train/validation object counts disagree with the split manifest")
    return record, source, specs, groups


def run(catalog_path: Path, output: Path, *, splits_path: Path | None = None,
        seed: int = 20261010, size: int = 384) -> dict:
    catalog_record, source, specs, assignments = _load_catalog(catalog_path, splits_path)
    if output.exists():
        raise ValueError(f"Benchmark output already exists: {output}")
    output.mkdir(parents=True)
    cases_dir, objects_dir, preflight_dir = (output / name for name in ("cases", "objects", "preflight"))
    cases_dir.mkdir()
    objects_dir.mkdir()
    preflight_dir.mkdir()
    views = load_views(REPO_ROOT / "configs/views.yaml")
    render = RenderConfig(REPO_ROOT / "web/dist", REPO_ROOT / "web/node_modules/.bin/esbuild", size, size)
    axis = get_corruption("kinematics.joint_axis", min_deg=45.0, max_deg=45.0)
    offset = get_corruption("structure.part_offset", min_m=0.05, max_m=0.05)
    cases, gold, objects = [], {split: [] for split in sorted(set(assignments.values()))}, []
    manifest = {"version": 1, "status": "building", "created_at": now_iso(),
                "verifier_schema_version": VERIFIER_SCHEMA_VERSION, "seed": seed,
                "dataset": "breezexian/UniPhys-Bench-Part2",
                "dataset_revision": catalog_record["dataset_revision"],
                "catalog": str(catalog_path),
                "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                "splits": str(splits_path) if splits_path else None,
                "splits_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest() if splits_path else None,
                "source_id": source.source_id,
                "constraints": {"max_triangles": MAX_TRIANGLES, "max_draw_calls": MAX_DRAW_CALLS},
                "git": git_state(REPO_ROOT), "objects": objects, "cases": cases}
    write_json_atomic(output / "manifest.json", manifest)
    try:
        for index, spec in enumerate(specs):
            object_id = spec.asset_id
            split_group = source.metadata(object_id)["split_group"]
            split = assignments[split_group]
            clean = ObjectProgram.from_spec(spec)
            if not axis.applicable(clean):
                raise ValueError(f"Selected object has no finite joint: {object_id}")
            axis_program, axis_label = axis.apply(clean, np.random.default_rng(seed + 2 * index))
            offset_program, offset_label = offset.apply(clean, np.random.default_rng(seed + 2 * index + 1))
            object_dir = objects_dir / object_id
            object_dir.mkdir()
            clean_path = object_dir / "clean.ts"
            clean_path.write_text(clean.source, encoding="utf-8")
            with ObservationSession(render, views) as session:
                session.load(clean, framing=None)
                image_path = object_dir / "reference.png"
                image_path.write_bytes(session.request_view("oblique_fr").png)
            objects.append({"id": object_id, "name": object_id,
                            "category": spec.category, "reference_sha": clean.sha,
                            "split_group": split_group, "split": split})

            variants = (("clean", clean, None), ("axis", axis_program, axis_label),
                        ("offset", offset_program, offset_label))
            for variant, program, label in variants:
                case_id = f"{object_id}-{variant}"
                program_path = clean_path if variant == "clean" else cases_dir / f"{case_id}.ts"
                if variant != "clean":
                    program_path.write_text(program.source, encoding="utf-8")
                finding = [] if label is None else [{
                    "defect_id": label.defect_id, "parts": list(label.parts),
                    "joint_id": label.params.get("joint") if label.defect_id == "kinematics.joint_axis" else None,
                }]
                validation = validate_code(program, clean, render, views, MAX_TRIANGLES, MAX_DRAW_CALLS)
                expected_correct = label is None
                if validation.get("status") != "complete" or validation.get("code_correct") != expected_correct:
                    raise ValueError(f"GT validation failed for {case_id}: {validation}")
                write_json_atomic(preflight_dir / f"{case_id}.json", validation)
                cases.append({"id": case_id, "program": program_path.relative_to(output).as_posix(),
                              "reference_program": clean_path.relative_to(output).as_posix(),
                              "reference_images": [image_path.relative_to(output).as_posix()],
                              "split_group": split_group, "split": split,
                              "task": "Verify the articulated candidate against the reference image."})
                gold[split].append({
                    "id": case_id, "program_sha": program.sha, "reference_sha": clean.sha,
                    "split_group": split_group, "split": split,
                    "target": {"status": "pass" if expected_correct else "fail", "findings": finding},
                })
            write_json_atomic(output / "manifest.json", manifest)
        _validate_case_identity(cases, gold)
        for split, rows in gold.items():
            (output / f"{split}.gold.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        object_counts = {split: sum(row["split"] == split for row in objects) for split in gold}
        case_counts = {split: len(rows) for split, rows in gold.items()}
        manifest.update(status="complete", object_count=len(objects), case_count=len(cases),
                        object_counts=object_counts, case_counts=case_counts,
                        preflight={"status": "complete", "validated_cases": len(cases)},
                        finished_at=now_iso())
        write_json_atomic(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc), finished_at=now_iso())
        write_json_atomic(output / "manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path,
                        default=Path("data/assets/uniphys-catalog-v1/catalog.json"))
    parser.add_argument("--splits", type=Path,
                        help="train/validation split manifest; omitted for the frozen test catalog")
    parser.add_argument("--out", type=Path,
                        default=Path("data/benchmarks/uniphys-verifier-60-catalog-v1"))
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()
    result = run(args.catalog.resolve(), args.out.resolve(),
                 splits_path=args.splits.resolve() if args.splits else None,
                 seed=args.seed, size=args.size)
    print(json.dumps({"status": result["status"], "objects": result["object_count"],
                      "cases": result["case_count"], "output": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
