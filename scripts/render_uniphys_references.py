#!/usr/bin/env python3
"""Render UniPhys objects into TurnItOver reference-view galleries.

Continuous joints have no finite upper limit, so this static-reference export
fixes them at their URDF zero pose. Revolute and prismatic joints retain their
limits and receive the normal ``export_preview`` upper-limit screenshot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.assets.urdf import UrdfModel
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.preview import export_preview
from turnitover.render.session import RenderConfig
from turnitover.render.views import load_views
from turnitover.telemetry import git_state, now_iso


def selected_ids(value: str) -> tuple[str, ...]:
    result = []
    for item in value.split(","):
        raw = item.strip()
        if raw.startswith("UPB_"):
            suffix = raw[4:]
        else:
            suffix = raw
        if len(suffix) > 8 or not suffix.isascii() or not suffix.isdecimal():
            raise ValueError(f"Invalid UniPhys object ID: {raw}")
        object_id = f"UPB_{int(suffix):08d}"
        if object_id not in result:
            result.append(object_id)
    if not result:
        raise ValueError("Select at least one UniPhys object")
    return tuple(result)


def load_download(root: Path, ids: tuple[str, ...]) -> tuple[dict, dict[str, dict]]:
    manifest_path = root / "download-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "download" or not manifest.get("revision"):
        raise ValueError("Expected a completed download manifest with an immutable revision")
    records = {row.get("object_id"): row for row in manifest.get("results", [])}
    for object_id in ids:
        row = records.get(object_id)
        if row is None or row.get("status") != "complete":
            raise ValueError(f"Object is not complete in download manifest: {object_id}")
    return manifest, records


def run(root: Path, output: Path, ids: tuple[str, ...], *, size: int) -> dict:
    if size <= 0:
        raise ValueError("--size must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    repo = REPO_ROOT
    manifest, downloads = load_download(root, ids)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": 1,
        "status": "running",
        "created_at": now_iso(),
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "view_config": str((repo / "configs/views.yaml").resolve()),
        "image_size": size,
        "git": git_state(repo),
        "objects": [],
    }
    summary_path = output / "manifest.json"

    def save() -> None:
        write_json_atomic(summary_path, summary)

    save()
    render = RenderConfig(repo / "web/dist", repo / "web/node_modules/.bin/esbuild", size, size)
    views = load_views(repo / "configs/views.yaml")
    try:
        for object_id in ids:
            folder = root / "data" / object_id
            metadata = json.loads((folder / "meta_data.json").read_text(encoding="utf-8"))
            model = UrdfModel(folder / "model.urdf", continuous_joint_policy="fixed")
            spec = model.spec(object_id, metadata["object"]["category"])
            program = ObjectProgram.from_spec(spec)
            target = output / object_id
            export_preview(program, render, views, target, videos=False)
            preview = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            rest = [row for row in preview["screenshots"] if row.get("joint_state") == "rest"]
            joints = [row for row in preview["screenshots"] if "joint_id" in row]
            summary["objects"].append({
                "object_id": object_id,
                "object_name": metadata["object"]["object_name"],
                "source": metadata["source"],
                "download_files": downloads[object_id]["files"],
                "program_sha": program.sha,
                "parts": len(spec.parts),
                "finite_joints": [joint.id for joint in spec.joints],
                "continuous_joints_fixed_at_zero": list(model.continuous_joints_fixed_at_zero),
                "static_views": len(rest),
                "joint_views": len(joints),
                "preview_manifest": f"{object_id}/manifest.json",
            })
            save()
        summary["status"] = "complete"
    except Exception as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        summary["finished_at"] = now_iso()
        save()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/assets/uniphys"))
    parser.add_argument("--out", type=Path, default=Path("data/references/uniphys"))
    parser.add_argument("--ids", default=",".join(str(i) for i in range(20)))
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.out.resolve(), selected_ids(args.ids), size=args.size)
    print({"status": result["status"], "objects": len(result["objects"]), "manifest": args.out / "manifest.json"})
