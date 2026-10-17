#!/usr/bin/env python3
"""Replay Level 0 with fixed Chromium evidence; it never calls a diagnosis model.

The private replay bundle supplies only executable programs and a pre-frozen
action/camera recipe.  Labels are deliberately copied only after all images
are rendered, for a separate scorer to consume.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from turnitover.core.actions import Detent, detent_value
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import from_dict
from turnitover.core.spec import AssetSpec
from turnitover.core.sample import Framing
from turnitover.render.session import ObservationSession, RenderConfig
from turnitover.render.views import load_views


TILE = 256
GAP = 10
SHEET = 542
SOURCE_COMMIT = "f34b07c127266b401564592bfe38bb6c9e7121fb"


def sha256(value: bytes | Path) -> str:
    data = value.read_bytes() if isinstance(value, Path) else value
    return hashlib.sha256(data).hexdigest()


def opaque_id(asset_id: str, kind: str, seed: int | None) -> str:
    digest = hashlib.sha256(f"level0-axis-v1:{asset_id}:{kind}:{seed}".encode()).hexdigest()
    return f"l0-{digest[:12]}"


def selected_seed(asset: dict) -> int:
    rows = [row["seed"] for row in asset["attempts"] if row.get("selected")]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one selected candidate for {asset['asset_id']}")
    return int(rows[0])


def load_program(path: Path) -> ObjectProgram:
    return ObjectProgram.from_spec(from_dict(AssetSpec, json.loads(path.read_text())))


def png_info(png: bytes) -> dict:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    if image.size != (TILE, TILE):
        raise ValueError(f"Expected {TILE}px tile, got {image.size}")
    return {"sha256": sha256(png), "size": list(image.size)}


def compose(tiles: list[bytes]) -> bytes:
    if len(tiles) != 4:
        raise ValueError("Need exactly ref-rest, ref-limit, candidate-rest, candidate-limit")
    sheet = Image.new("RGB", (SHEET, SHEET), "#fafaf8")
    for (x, y), tile in zip(((GAP, GAP), (276, GAP), (GAP, 276), (276, 276)), tiles):
        sheet.paste(Image.open(io.BytesIO(tile)).convert("RGB"), (x, y))
    output = io.BytesIO(); sheet.save(output, format="PNG")
    return output.getvalue()


def pixel_difference(first: bytes, second: bytes) -> dict:
    a = np.asarray(Image.open(io.BytesIO(first)).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(io.BytesIO(second)).convert("RGB"), dtype=np.int16)
    delta = np.abs(a - b)
    return {"mean_abs_rgb": float(delta.mean()), "max_abs_rgb": int(delta.max()), "byte_equal": first == second}


def observe(session: ObservationSession, program: ObjectProgram, framing: Framing, view: str, joint: str) -> tuple[list[bytes], dict]:
    """The fixed policy: reset/rest observation, then one limit actuation/observation."""
    info = session.load(program, framing)
    try:
        if joint not in info.joints:
            raise ValueError(f"Fixed joint {joint} absent")
        session.reset_joints()
        rest = session.request_view(view)
        limit = session.actuate(joint, Detent.LIMIT)
        after = session.request_view(view)
        expected_limit = detent_value(Detent.LIMIT, tuple(info.joints[joint]["limits"]))
        if limit.value != expected_limit:
            raise ValueError(f"Limit result mismatch: {limit.value} != {expected_limit}")
        return [rest.png, after.png], {
            "fixed_actions": ["reset_joints", "request_view(rest)", "actuate_joint(limit)", "request_view(limit)"],
            "view": view, "joint": joint, "joint_limits": list(info.joints[joint]["limits"]),
            "limit_value": limit.value, "rest_camera": rest.camera, "limit_render_ms": after.render_ms,
        }
    finally:
        session.dispose()


def main(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError("Output must be fresh")
    screening = json.loads((args.private_source / "screening.json").read_text())
    source_manifest = json.loads((args.source_dataset / "manifest.json").read_text())
    views = load_views(args.views)
    cfg = RenderConfig(args.harness, args.esbuild, width=TILE, height=TILE)
    args.out.mkdir(parents=True)
    inputs = args.out / "inputs"; inputs.mkdir()
    evidence = []
    with ObservationSession(cfg, views) as session:
        environment = {"browser_version": session.browser_version(), "gl": session.gl_info(),
                       "harness_dist_sha256": sha256(args.harness / "index.html"),
                       "views_sha256": sha256(args.views), "render_size": [TILE, TILE]}
        for asset in screening["assets"]:
            asset_id, joint, view = asset["asset_id"], asset["joint_id"], asset["view"]
            framing = Framing(center=tuple(asset["normal_framing"]["center"]), radius=float(asset["normal_framing"]["radius"]))
            axis_seed = selected_seed(asset)
            for kind, case_id in (("clean", opaque_id(asset_id, "clean", None)), ("joint_axis", opaque_id(asset_id, "joint_axis", axis_seed))):
                # Program identity and fixed recipe are private execution inputs, never model text.
                program = load_program(args.private_source / f"{case_id}.json")
                reference = load_program(args.private_source / f"{opaque_id(asset_id, 'clean', None)}.json")
                ref_tiles, ref_trace = observe(session, reference, framing, view, joint)
                cand_tiles, cand_trace = observe(session, program, framing, view, joint)
                if ref_trace["rest_camera"] != cand_trace["rest_camera"]:
                    raise ValueError(f"Camera drift between reference/candidate for {case_id}")
                rendered = compose([ref_tiles[0], ref_tiles[1], cand_tiles[0], cand_tiles[1]])
                image_path = inputs / f"{case_id}.png"; image_path.write_bytes(rendered)
                original = args.source_dataset / "inputs" / f"{case_id}.png"
                if sha256(original) != source_manifest["files"][f"inputs/{case_id}.png"]:
                    raise ValueError(f"Original source SHA mismatch: {case_id}")
                evidence.append({"case_id": case_id, "image": f"inputs/{case_id}.png", "sheet_size": [SHEET, SHEET],
                                 "position_order": ["ref-rest", "ref-limit", "candidate-rest", "candidate-limit"],
                                 "tile_info": [png_info(tile) for tile in [ref_tiles[0], ref_tiles[1], cand_tiles[0], cand_tiles[1]]],
                                 "reference_trace": ref_trace, "candidate_trace": cand_trace,
                                 "original_comparison": pixel_difference(rendered, original.read_bytes())})
    # Labels are copied only after all fixed-policy evidence exists; the renderer never reads them.
    shutil.copyfile(args.source_dataset / "human_review.jsonl", args.out / "human_review.jsonl")
    (args.out / "replay_manifest.json").write_text(json.dumps({"status": "complete", "source_commit": SOURCE_COMMIT,
        "model_visible_fields": ["image", "fixed binary prompt"], "not_model_visible": ["asset_id", "joint_id", "framing", "candidate parameters", "gold"],
        "environment": environment, "cases": evidence}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--private-source", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--esbuild", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
