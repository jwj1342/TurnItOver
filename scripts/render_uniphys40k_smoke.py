#!/usr/bin/env python3
"""Render one converted UniPhys-40K asset and verify each browser-side joint motion."""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageChops

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.assets.catalog import CatalogSource
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.render.session import ObservationSession, RenderConfig
from turnitover.render.views import ViewDef
from turnitover.core.actions import ViewId


def _descendants(spec, root: str) -> set[str]:
    result = {root}
    changed = True
    while changed:
        changed = False
        for part in spec.parts:
            if part.parent in result and part.id not in result:
                result.add(part.id)
                changed = True
    return result


def run(catalog: Path, output: Path, asset_id: str, *, size: int = 512) -> Path:
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    source = CatalogSource(str(catalog))
    spec = source.get(asset_id)
    if not spec.joints:
        raise ValueError("Smoke test requires at least one joint")
    output.mkdir(parents=True)
    views = (
        ViewDef(ViewId("iso"), 35, 20, "perspective", 1.25),
        ViewDef(ViewId("opposite"), 215, 20, "perspective", 1.25),
        ViewDef(ViewId("side"), 125, 10, "perspective", 1.25),
    )
    render = RenderConfig(REPO_ROOT / "web/dist", REPO_ROOT / "web/node_modules/.bin/esbuild",
                          size, size)
    program = ObjectProgram.from_spec(spec)
    joint_results = []
    with ObservationSession(render, views) as session:
        for joint in spec.joints:
            moving = _descendants(spec, joint.part)
            info = session.load(program, framing=None)
            rest_pngs = {str(view.id): session.request_view(view.id).png for view in views}
            rest_geometry = session.export_geometry()
            session.actuate_joint(joint.id, joint.limits[1])
            upper_pngs = {str(view.id): session.request_view(view.id).png for view in views}
            upper_geometry = session.export_geometry()
            changed_pixels = {}
            prefix = joint.id.removeprefix("joint_")
            for view_id in rest_pngs:
                (output / f"{prefix}-{view_id}-rest.png").write_bytes(rest_pngs[view_id])
                (output / f"{prefix}-{view_id}-upper.png").write_bytes(upper_pngs[view_id])
                with Image.open(io.BytesIO(rest_pngs[view_id])) as left, \
                        Image.open(io.BytesIO(upper_pngs[view_id])) as right:
                    difference = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
                    changed_pixels[view_id] = int(
                        np.count_nonzero(np.any(np.asarray(difference) != 0, axis=2)))
            if not joint_results:
                (output / "rest.png").write_bytes(rest_pngs["iso"])
                (output / "upper.png").write_bytes(upper_pngs["iso"])
            displacement = {
                part_id: float(np.max(np.linalg.norm(upper_geometry[part_id].vertices
                                                 - rest_geometry[part_id].vertices, axis=1)))
                for part_id in rest_geometry
            }
            static = set(displacement) - moving
            if not all(value > 0 for value in changed_pixels.values()) or not all(
                    displacement[part_id] > 1e-5 for part_id in moving):
                raise RuntimeError(
                    f"joint {joint.id} did not visibly and geometrically move every dependency part")
            if not all(displacement[part_id] < 1e-6 for part_id in static):
                raise RuntimeError(f"joint {joint.id} moved a part outside the dependency group")
            joint_results.append({
                "id": joint.id, "upper": joint.limits[1],
                "moving_parts": sorted(moving), "static_parts": sorted(static),
                "max_vertex_displacement_m": displacement,
                "changed_pixels_by_view": changed_pixels,
                "render_sha256": {
                    view_id: {"rest": hashlib.sha256(rest_pngs[view_id]).hexdigest(),
                              "upper": hashlib.sha256(upper_pngs[view_id]).hexdigest()}
                    for view_id in rest_pngs
                },
            })
        browser = session.browser_version()
        gl = session.gl_info()
    report = {
        "version": 1, "status": "mechanical_checks_complete", "asset_id": asset_id,
        "catalog": str(catalog.resolve()), "program_sha": program.sha,
        "joints": joint_results,
        "range_mapping": "source_qpos_relative_to_released_obj_zero_pose",
        "physical_motion_review": "required",
        "framing": {"center": list(info.framing.center), "radius": info.framing.radius},
        "browser": browser, "gl": gl,
    }
    if len(joint_results) == 1:
        legacy = joint_results[0]
        report.update(joint={"id": legacy["id"], "upper": legacy["upper"]},
                      moving_parts=legacy["moving_parts"], static_parts=legacy["static_parts"],
                      max_vertex_displacement_m=legacy["max_vertex_displacement_m"],
                      changed_pixels_by_view=legacy["changed_pixels_by_view"],
                      render_sha256=legacy["render_sha256"])
    write_json_atomic(output / "smoke.json", report)
    return output / "smoke.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path,
                        default=Path("data/assets/uniphys40k-pilot-catalog/catalog.json"))
    parser.add_argument("--asset-id", default="UP_00023000")
    parser.add_argument("--out", type=Path, default=Path("output/uniphys40k-pilot-smoke"))
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    print(run(args.catalog.resolve(), args.out.resolve(), args.asset_id, size=args.size))


if __name__ == "__main__":
    main()
