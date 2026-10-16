"""Auditor-side sweep of how visible a joint_axis corruption is across the teacher's action space.

For each frozen-witness case the clean reference spec is rebuilt from the catalog and the
candidate is re-derived from the recorded corruption label (both must reproduce the gold
program SHAs). The current corruption is then rendered against the reference at the same
state for every allowed view and every reachable combination of joint detents; a strength
gradient re-rotates the same joint axis about the same perpendicular by other angles, with
object, framing, views, states and resolution held fixed. Per pair we record pixel change,
eroded core change, the largest connected changed region, and the actual world-space and
projected screen-space vertex displacement exported by the browser. Pixel metrics are a
proxy: semantic identifiability is judged separately from the saved images. Nothing here
calls a model API or touches frozen data; all output goes under --out.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from turnitover.assets.catalog import CatalogSource
from turnitover.config import load_generate_config
from turnitover.core.actions import Detent, detent_value
from turnitover.core.program import ObjectProgram
from turnitover.oracle.evidence import ExposureThresholds, pixels
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.storage.reader import iter_samples
from turnitover.teacher.witnesses import (PERCEPTIBLE_CORE_FRACTION, clamp, core_changed_fraction, rest_state,
                                          same_state)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/benchmarks/partnet-mobility-formal-v1/prepared"
DEFAULT_CONFIG = ROOT / "configs/generate_partnet_mobility_supervision.yaml"
DEFAULT_WITNESSES = ROOT / "output/interactive-teacher/axis-round4-frozen-witnesses"
DETENTS = (Detent.ZERO, Detent.MID, Detent.LIMIT)
DEG = math.pi / 180


def _rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}


def _raw_sample(raw_root: Path, source_id: str, sample_id: str):
    for tar_path in sorted(raw_root.glob("*.tar")):
        for sample, _ in iter_samples(tar_path):
            if sample.source_id == source_id and sample.sample_id == sample_id:
                return sample
    raise ValueError(f"No raw sample {sample_id}")


def axis_at(original, corrupted, angle_deg: float) -> tuple[float, float, float]:
    """Rotate the original axis toward the recorded corruption's perpendicular by angle_deg."""
    a = np.asarray(original, float) / np.linalg.norm(original)
    b = np.asarray(corrupted, float)
    theta = math.acos(float(np.clip(a.dot(b), -1, 1)))
    perp = (b - a * math.cos(theta)) / math.sin(theta)
    new = a * math.cos(angle_deg * DEG) + perp * math.sin(angle_deg * DEG)
    return tuple(float(x) for x in new / np.linalg.norm(new))


def joint_states(limits: dict, target: str) -> list[tuple[str, dict]]:
    """Every reachable combination of grid detents (other joints also at rest/zero/mid/limit)."""
    def values(joint):
        bounds = limits[joint]
        unique = {}
        for name, value in [("rest", clamp(0.0, bounds))] + [
                (d.value, clamp(detent_value(d, bounds), bounds)) for d in DETENTS]:
            unique.setdefault(round(value, 9), (name, value))  # rest and zero coincide when lo == 0
        return list(unique.values())
    joints = sorted(limits)
    return [(",".join(f"{j}={n}" for j, (n, _) in zip(joints, combo)),
             {j: v for j, (_, v) in zip(joints, combo)})
            for combo in itertools.product(*(values(j) for j in joints))]


def project(vertices: np.ndarray, view, framing, width: int, height: int) -> np.ndarray:
    """Pixel coordinates of world vertices for a view, mirroring web/src/camera.ts."""
    az, el = view.azimuth_deg * DEG, view.elevation_deg * DEG
    direction = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
    center, radius, aspect = np.asarray(framing.center, float), framing.radius, width / height
    if view.projection == "orthographic":
        eye = center + direction * radius * 3
    else:
        eye = center + direction * (radius * view.distance_factor
                                    / math.sin(20 * DEG) * 1.05)
    up = np.array([0.0, 0.0, -1.0]) if abs(view.elevation_deg) > 89 else np.array([0.0, 1.0, 0.0])
    z = eye - center
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rel = vertices - eye
    cx, cy, cz = rel @ x, rel @ y, rel @ z
    if view.projection == "orthographic":
        h = radius * 1.1
        nx, ny = cx / (h * aspect), cy / h
    else:
        f = 1 / math.tan(20 * DEG)
        nx, ny = cx * f / (-cz * aspect), cy * f / -cz
    return np.stack([(nx + 1) * width / 2, (1 - ny) * height / 2], axis=1)


def pair_metrics(cand: np.ndarray, ref: np.ndarray) -> dict:
    mask = np.abs(cand.astype(np.float32) - ref.astype(np.float32)).max(axis=2) / 255. > ExposureThresholds().pixel_delta
    eroded = ndimage.binary_erosion(mask, structure=np.ones((5, 5)), border_value=0)
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:] if count else np.array([0])
    core_labels, core_count = ndimage.label(eroded)
    core_sizes = np.bincount(core_labels.ravel())[1:] if core_count else np.array([0])
    box = None
    if count:
        ys, xs = np.nonzero(labels == int(np.argmax(sizes)) + 1)
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    core = core_changed_fraction(cand, ref)
    return {"changed_fraction": float(mask.mean()), "core_changed_fraction": core,
            "largest_region_px": int(sizes.max()), "largest_core_region_px": int(core_sizes.max()),
            "largest_region_box_xyxy": box, "perceptible_rule": core >= PERCEPTIBLE_CORE_FRACTION}


def displacement(cand_geo: dict, ref_geo: dict, view, framing, size) -> dict:
    world, screen, parts = 0.0, 0.0, {}
    for name, mesh in ref_geo.items():
        other = cand_geo.get(name)
        if other is None or other.vertices.shape != mesh.vertices.shape:
            raise ValueError(f"Part {name} topology differs between candidate and reference")
        d = np.linalg.norm(other.vertices - mesh.vertices, axis=1)
        if d.max() <= 1e-9:
            continue
        px = np.linalg.norm(project(other.vertices, view, framing, *size) - project(mesh.vertices, view, framing, *size),
                            axis=1)
        parts[name] = {"world_max": float(d.max()), "world_mean": float(d.mean()),
                       "px_max": float(px.max()), "px_p95": float(np.percentile(px, 95))}
        world, screen = max(world, float(d.max())), max(screen, float(px.max()))
    return {"world_max": world, "px_max": screen, "parts": parts}


class Renderer:
    """One loaded program per call; every state then renders all requested views."""

    def __init__(self, session, framing, views, size):
        self.session, self.framing, self.views, self.size = session, framing, views, size

    def run(self, program: ObjectProgram, states: list[dict], view_ids: list[str]):
        self.session.load(program, framing=self.framing)
        try:
            out = []
            for state in states:
                self.session.set_joints(state)
                geometry = self.session.export_geometry()
                out.append((geometry, {v: self.session.request_view(v).png for v in view_ids}))
            return out
        finally:
            self.session.dispose()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_case(uid: str, args, session, views_by_id, gold, catalog, size) -> dict:
    manifest = json.loads((args.frozen_witnesses / uid / "manifest.json").read_text())
    row = gold[uid]
    sample = _raw_sample(args.benchmark.parent / "raw", row["source_id"], row["sample_id"])
    (corruption,) = [c for c in sample.corruptions if c.defect_id == "kinematics.joint_axis"]
    joint_id = corruption.params["joint"]
    spec = catalog.get(row["asset_id"])
    reference = ObjectProgram.from_spec(spec)
    candidate = ObjectProgram.from_spec(spec.replace_joint(joint_id, axis=tuple(corruption.label["axis"])))
    if reference.sha != row["reference_sha"] or candidate.sha != row["program_sha"]:
        raise ValueError(f"{uid}: rebuilt programs do not reproduce gold SHAs")
    original = tuple(corruption.params["original_axis"])
    actual_angle = float(corruption.params["angle_deg"])
    limits = {j: tuple(v) for j, v in manifest["candidate_joint_limits"].items()}
    joint_type = spec.joint(joint_id).type.value
    renderer = Renderer(session, sample.framing, views_by_id, size)
    case_dir = args.out / uid
    (case_dir / "png").mkdir(parents=True, exist_ok=True)
    states = joint_states(limits, joint_id)
    view_ids = list(views_by_id)

    def save(png: bytes, name: str) -> str:
        path = case_dir / "png" / name
        path.write_bytes(png)
        return path.relative_to(args.out).as_posix()

    ref_renders = renderer.run(reference, [s for _, s in states], view_ids)
    rest = rest_state(limits)
    rest_index = next(i for i, (_, s) in enumerate(states) if same_state(s, rest))

    def compare(program, tag):
        rows = []
        for (label, state), (geo, pngs), (ref_geo, ref_pngs) in zip(states, renderer.run(program, [s for _, s in states], view_ids), ref_renders):
            for view in view_ids:
                cand, ref = pixels(pngs[view]), pixels(ref_pngs[view])
                motion = pair_metrics(ref, pixels(ref_renders[rest_index][1][view]))
                stem = f"{view}__{label.replace(',', '__').replace('=', '-')}"
                rows.append({"view": view, "state_label": label, "state": state,
                             **pair_metrics(cand, ref),
                             "reference_motion_changed_fraction": motion["changed_fraction"],
                             "reference_motion_core_fraction": motion["core_changed_fraction"],
                             "displacement": displacement(geo, ref_geo, views_by_id[view], sample.framing, size),
                             "candidate_png": save(pngs[view], f"{tag}__{stem}.png"),
                             "candidate_sha256": sha(pngs[view]),
                             "reference_png": save(ref_pngs[view], f"reference__{stem}.png"),
                             "reference_sha256": sha(ref_pngs[view])})
        return rows

    current = compare(candidate, f"angle{actual_angle:.2f}")
    # The fresh one-load renders must agree with the frozen witnesses wherever states coincide.
    verified = 0
    for witness in manifest["witnesses"]:
        match = [r for r in current if r["view"] == witness["view"] and same_state(r["state"], witness["state"])]
        if not match or match[0]["candidate_sha256"] != witness["sha256"]:
            raise ValueError(f"{uid}: sweep render differs from frozen witness {witness['index']}")
        verified += 1
    gradient = []
    for angle in args.angles:
        program = ObjectProgram.from_spec(spec.replace_joint(joint_id, axis=axis_at(original, corruption.label["axis"], angle)))
        rows = current if abs(angle - actual_angle) < 1e-9 else compare(program, f"angle{angle:.2f}")
        gradient.append({"angle_deg": angle, "program_sha": program.sha,
                         "rows": [r for r in rows if r["view"] in args.gradient_views]})
    print(json.dumps({"id": uid, "states": len(states), "views": len(view_ids), "verified_witnesses": verified}),
          flush=True)
    return {"id": uid, "asset_id": row["asset_id"], "category": row["category"], "sample_id": row["sample_id"],
            "joint": joint_id, "joint_type": joint_type, "joint_limits": {k: list(v) for k, v in limits.items()},
            "original_axis": list(original), "corrupted_axis": list(corruption.label["axis"]),
            "actual_angle_deg": actual_angle, "framing": {"center": list(sample.framing.center),
                                                           "radius": sample.framing.radius},
            "candidate_sha": candidate.sha, "reference_sha": reference.sha,
            "frozen_witnesses_verified": verified, "current": current,
            "gradient_views": list(args.gradient_views), "gradient": gradient}


def main(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--angles", type=float, nargs="+", default=[5, 15, 30, 45, 60, 90])
    parser.add_argument("--gradient-views", nargs="+", default=["front", "oblique_fr"])
    parser.add_argument("--frozen-witnesses", type=Path, default=DEFAULT_WITNESSES)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = load_generate_config(args.config)
    views = load_views(cfg.views_path)
    gold = _rows(args.benchmark / "train.gold.jsonl")
    catalog = CatalogSource(cfg.asset_source["manifest"])
    size = (cfg.render.width, cfg.render.height)
    results = []
    with ObservationSession(cfg.render, views) as session:
        for uid in args.sample_id:
            results.append(run_case(uid, args, session, {str(v.id): v for v in views}, gold, catalog, size))
    (args.out / "summary.json").write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
