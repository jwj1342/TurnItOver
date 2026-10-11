#!/usr/bin/env python3
"""Build 20 UniPhys objects x clean/axis/offset as a frozen verifier benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.assets.urdf import UrdfModel
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.corruptions import get_corruption
from turnitover.render.session import ObservationSession, RenderConfig
from turnitover.render.views import load_views
from turnitover.telemetry import git_state, now_iso
from turnitover.verifier.contracts import VERIFIER_SCHEMA_VERSION
from turnitover.verifier.gt_validator import validate_code


OBJECT_IDS = (
    "UPB_00000000", "UPB_00000002", "UPB_00000004", "UPB_00000006",
    "UPB_00000007", "UPB_00000009", "UPB_00000010", "UPB_00000012",
    "UPB_00000014", "UPB_00000015", "UPB_00000016", "UPB_00000017",
    "UPB_00000018", "UPB_00000019", "UPB_00000020", "UPB_00000021",
    "UPB_00000022", "UPB_00000025", "UPB_00000026", "UPB_00000027",
)
MAX_TRIANGLES = 1_000_000
MAX_DRAW_CALLS = 1_024


def run(root: Path, output: Path, *, seed: int = 20261010, size: int = 384) -> dict:
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
    cases, gold, objects = [], [], []
    manifest = {"version": 1, "status": "building", "created_at": now_iso(),
                "verifier_schema_version": VERIFIER_SCHEMA_VERSION, "seed": seed,
                "dataset": "breezexian/UniPhys-Bench-Part2",
                "dataset_revision": "ea633b5c2b1cf9fefb24fb8c30c3703cc75ae7c4",
                "constraints": {"max_triangles": MAX_TRIANGLES, "max_draw_calls": MAX_DRAW_CALLS},
                "git": git_state(REPO_ROOT), "objects": objects, "cases": cases}
    write_json_atomic(output / "manifest.json", manifest)
    try:
        for index, object_id in enumerate(OBJECT_IDS):
            folder = root / "data" / object_id
            metadata = json.loads((folder / "meta_data.json").read_text(encoding="utf-8"))
            spec = UrdfModel(folder / "model.urdf", continuous_joint_policy="fixed").spec(
                object_id, metadata["object"]["category"])
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
            objects.append({"id": object_id, "name": metadata["object"]["object_name"],
                            "category": metadata["object"]["category"], "reference_sha": clean.sha})

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
                              "task": "Verify the articulated candidate against the reference image."})
                gold.append({"id": case_id, "program_sha": program.sha, "reference_sha": clean.sha,
                             "target": {"status": "pass" if expected_correct else "fail",
                                        "findings": finding}})
            write_json_atomic(output / "manifest.json", manifest)
        (output / "test.gold.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in gold), encoding="utf-8")
        manifest.update(status="complete", object_count=len(objects), case_count=len(cases),
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
    parser.add_argument("--root", type=Path, default=Path("data/assets/uniphys"))
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/uniphys-verifier-60"))
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.out.resolve(), seed=args.seed, size=args.size)
    print(json.dumps({"status": result["status"], "objects": result["object_count"],
                      "cases": result["case_count"], "output": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
