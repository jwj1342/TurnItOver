"""Compose the evidence figures of the corruption-perceptibility audit from saved renders only.

Every tile is a byte-identical PNG written by ``sweep_corruption_perceptibility`` or a saved
teacher observation; this script only lays tiles out, draws a difference heatmap, crops a
nearest-neighbour zoom of the largest changed region, and records source paths plus
SHA-256 for every input and output in ``figures.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from turnitover.oracle.evidence import ExposureThresholds


ROOT = Path(__file__).resolve().parents[1]
TILE, GAP, HEADER = 256, 6, 34
SURFACE, INK, INK_2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)
ZOOM_BOX = (235, 104, 52)  # categorical slot 2; identifies the zoom region, not a status
# Sequential blue ramp (steps 100..700) for |difference| magnitude; saturates at HEAT_MAX.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
HEAT_MAX = 0.25

IDS = {"PM_102080": "b124e209888e8e2e692ad38a", "PM_5088": "78daaf49b71ff5f5df8a1a29",
       "PM_103275": "ec3acb502c61937e9b968ab1", "PM_25308": "27d942cd3de2d333f24d6279"}

KEY_PANELS = [
    ("PM_102080", "oblique_fr", "joint_0=limit"),
    ("PM_102080", "front", "joint_0=limit"),
    ("PM_5088", "front", "joint_0=limit"),
    ("PM_5088", "front", "joint_0=zero"),
    ("PM_103275", "front", "joint_0=rest,joint_1=mid"),
    ("PM_103275", "front", "joint_0=rest,joint_1=limit"),
    ("PM_25308", "front", "joint_0=rest,joint_1=limit"),
    ("PM_25308", "front", "joint_0=rest,joint_1=mid"),
]
GRADIENTS = [
    ("PM_25308", "front", "joint_0=rest,joint_1=limit", [0, 5, 15, 30, 45, 60, 90, 180]),
    ("PM_102080", "oblique_fr", "joint_0=limit", [0, 30, 60, 90, 180]),
    ("PM_5088", "front", "joint_0=limit", [0, 5, 15, 30, 60]),
    ("PM_103275", "front", "joint_0=rest,joint_1=mid", [0, 5, 15, 30, 60]),
]
STATE_GRIDS = [
    ("PM_103275", "joint1-states-and-views", [
        ("front", "joint_0=rest,joint_1=zero"), ("front", "joint_0=rest,joint_1=mid"),
        ("front", "joint_0=rest,joint_1=limit"), ("oblique_fr", "joint_0=rest,joint_1=limit"),
        ("grazing_front", "joint_0=rest,joint_1=mid"), ("right", "joint_0=limit,joint_1=mid")]),
    ("PM_25308", "joint1-views", [
        ("front", "joint_0=rest,joint_1=mid"), ("front", "joint_0=rest,joint_1=limit"),
        ("oblique_fr", "joint_0=rest,joint_1=limit"), ("oblique_fl", "joint_0=rest,joint_1=limit"),
        ("top", "joint_0=rest,joint_1=limit")]),
    ("PM_102080", "all-views-limit", [(v, "joint_0=limit") for v in (
        "front", "back", "left", "right", "top", "bottom", "oblique_fr", "oblique_fl",
        "oblique_br", "oblique_bl", "grazing_front", "grazing_top")]),
]
# (case, episode dir, steps to show, reference-NN files shown to the teacher for comparison)
TEACHER_VIEWS = [
    ("PM_103275", "kimi-round4-active-HC-pm103275-axis-r2", [4, 5], ["reference-05.png", "reference-04.png"]),
    ("PM_25308", "kimi-round4-fixed-HC-pm25308-axis-rep2", [5, 6], ["reference-03.png", "reference-04.png"]),
    ("PM_5088", "kimi-round4-fixed-MH-pm5088-axis-rep2", [2, 4, 7, 9],
     ["reference-01.png", "reference-03.png", "reference-05.png", "reference-07.png"]),
]


def _font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT, SMALL = _font(15), _font(12)


def _hex(color: str) -> np.ndarray:
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], float)


def _action(decision: dict) -> str:
    if decision["type"] == "actuate_joint":
        return f"actuate {decision['joint_id']} -> {decision['detent']}"
    return f"request_view {decision['view_id']}" if decision["type"] == "request_view" else decision["type"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def heat(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """|difference| on the light surface: below the exposure threshold stays surface, then blue ramp."""
    d = np.abs(ref.astype(np.float32) - cand.astype(np.float32)).max(axis=2) / 255.
    ghost = ref.astype(np.float32).mean(axis=2, keepdims=True) / 255.
    out = np.asarray(SURFACE, float) * (1 - 0.18 * ghost)  # faint silhouette for orientation
    stops = np.stack([_hex(c) for c in RAMP])
    t = np.clip(d / HEAT_MAX, 0, 1) * (len(RAMP) - 1)
    lo = np.floor(t).astype(int).clip(0, len(RAMP) - 2)
    color = stops[lo] + (stops[lo + 1] - stops[lo]) * (t - lo)[..., None]
    mask = d > ExposureThresholds().pixel_delta
    out[mask] = color[mask]
    return out.astype(np.uint8)


def zoom_box(box, size=TILE, minimum=40, pad=8):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    side = max(minimum, x1 - x0 + 1 + 2 * pad, y1 - y0 + 1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(min(max(cx - side / 2, 0), size - side))
    top = int(min(max(cy - side / 2, 0), size - side))
    side = min(side, size)
    return max(left, 0), max(top, 0), max(left, 0) + side, max(top, 0) + side


def crop(img: np.ndarray, box) -> np.ndarray:
    if box is None:
        return np.full((TILE, TILE, 3), SURFACE, np.uint8)
    return np.asarray(Image.fromarray(img[box[1]:box[3], box[0]:box[2]]).resize((TILE, TILE), Image.NEAREST))


def outline(img: np.ndarray, box) -> np.ndarray:
    if box is None:
        return img
    pil = Image.fromarray(img.copy())
    ImageDraw.Draw(pil).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], outline=ZOOM_BOX, width=2)
    return np.asarray(pil)


def compose(rows: list[list[tuple[str, np.ndarray]]], title: str | None = None) -> Image.Image:
    cols = max(len(r) for r in rows)
    top = HEADER if title else 0
    width = cols * TILE + (cols + 1) * GAP
    height = top + len(rows) * (TILE + HEADER) + (len(rows) + 1) * GAP
    canvas = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((GAP, 8), title, fill=INK, font=FONT)
    for r, row in enumerate(rows):
        y = top + GAP + r * (TILE + HEADER + GAP)
        for c, (label, tile) in enumerate(row):
            x = GAP + c * (TILE + GAP)
            for i, line in enumerate(label.split("\n")[:2]):
                draw.text((x, y + 2 + i * 15), line, fill=INK if i == 0 else INK_2, font=SMALL)
            canvas.paste(Image.fromarray(tile), (x, y + HEADER))
    return canvas


class Builder:
    def __init__(self, sweep_path: Path, out: Path):
        self.sweep_root = sweep_path.parent
        self.cases = {c["asset_id"]: c for c in json.loads(sweep_path.read_text())}
        self.out = out
        self.records = []

    def row(self, asset, view, state, angle=None):
        case = self.cases[asset]
        if angle is None:
            rows = case["current"]
        else:
            (g,) = [g for g in case["gradient"] if abs(g["angle_deg"] - angle) < 1e-9]
            rows = g["rows"]
        (row,) = [r for r in rows if r["view"] == view and r["state_label"] == state]
        return row

    def images(self, row):
        ref_path, cand_path = self.sweep_root / row["reference_png"], self.sweep_root / row["candidate_png"]
        return ref_path, cand_path, load(ref_path), load(cand_path)

    def save(self, image: Image.Image, name: str, sources: list[Path], meta: dict):
        path = self.out / name
        image.save(path, optimize=True)
        self.records.append({"figure": path.resolve().relative_to(ROOT).as_posix(), "sha256": sha(path),
                             "sources": [{"path": p.resolve().relative_to(ROOT).as_posix(), "sha256": sha(p)}
                                         for p in sources], **meta})

    @staticmethod
    def metrics(row) -> str:
        return (f"chg {row['changed_fraction']:.4f}  core {row['core_changed_fraction']:.4f}  "
                f"region {row['largest_core_region_px']}px  disp {row['displacement']['px_max']:.1f}px")

    def key_panel(self, asset, view, state):
        row = self.row(asset, view, state)
        ref_path, cand_path, ref, cand = self.images(row)
        box = zoom_box(row["largest_region_box_xyxy"])
        angle = self.cases[asset]["actual_angle_deg"]
        tiles = [[(f"reference\n{view} | {state}", outline(ref, box)),
                  (f"candidate (current {angle:.1f} deg)\n{view} | {state}", outline(cand, box)),
                  ("|difference| heatmap\nsurface = below exposure threshold", heat(ref, cand)),
                  ("zoom: reference\norange box, nearest-neighbour", crop(ref, box)),
                  ("zoom: candidate\norange box, nearest-neighbour", crop(cand, box))]]
        name = f"{asset}__key__{view}__{state.replace(',', '__').replace('=', '-')}.png"
        self.save(compose(tiles, f"{asset}  {self.metrics(row)}"), name, [ref_path, cand_path],
                  {"kind": "key_panel", "asset": asset, "view": view, "state": state,
                   "angle_deg": angle, "metrics": {k: row[k] for k in (
                       "changed_fraction", "core_changed_fraction", "largest_region_px",
                       "largest_core_region_px", "perceptible_rule")},
                   "displacement": row["displacement"], "zoom_box_xyxy": box})

    def gradient(self, asset, view, state, angles):
        top, bottom, sources, metrics = [], [], [], []
        actual = self.cases[asset]["actual_angle_deg"]
        series = sorted([(a, a) for a in angles] + [(actual, None)])
        for angle, key in series:
            row = self.row(asset, view, state, key)
            ref_path, cand_path, ref, cand = self.images(row)
            tag = f"{angle:.1f} deg" + (" (current)" if key is None else "")
            top.append((f"candidate {tag}\ncore {row['core_changed_fraction']:.4f}", cand))
            bottom.append((f"|diff| {tag}\ndisp {row['displacement']['px_max']:.1f}px", heat(ref, cand)))
            sources += [ref_path, cand_path]
            metrics.append({"angle_deg": angle, "current": key is None, **{k: row[k] for k in (
                "changed_fraction", "core_changed_fraction", "largest_core_region_px", "perceptible_rule")},
                "px_max": row["displacement"]["px_max"], "world_max": row["displacement"]["world_max"]})
        name = f"{asset}__gradient__{view}__{state.replace(',', '__').replace('=', '-')}.png"
        ref_row = self.row(asset, view, state)
        ref_path = self.sweep_root / ref_row["reference_png"]
        tiles = [[("reference (same state)\n" + view, load(ref_path))] + top,
                 [("", np.full((TILE, TILE, 3), SURFACE, np.uint8))] + bottom]
        self.save(compose(tiles, f"{asset}  axis-error gradient  {view} | {state}"), name,
                  sorted(set(sources)), {"kind": "gradient", "asset": asset, "view": view, "state": state,
                                         "series": metrics})

    def state_grid(self, asset, tag, pairs):
        rows, sources, metrics = [], [], []
        for view, state in pairs:
            row = self.row(asset, view, state)
            ref_path, cand_path, ref, cand = self.images(row)
            sources += [ref_path, cand_path]
            rows.append([(f"reference\n{view} | {state}", ref), ("candidate (current)\n" + f"core {row['core_changed_fraction']:.4f}  region {row['largest_core_region_px']}px", cand),
                         ("|difference|\n" + f"disp {row['displacement']['px_max']:.1f}px", heat(ref, cand))])
            metrics.append({"view": view, "state": state, **{k: row[k] for k in (
                "changed_fraction", "core_changed_fraction", "largest_core_region_px", "perceptible_rule")},
                "px_max": row["displacement"]["px_max"]})
        if len(rows) > 6:  # wide grids: three (reference, candidate, diff) triples per line
            rows = [sum(rows[i:i + 3], []) for i in range(0, len(rows), 3)]
        self.save(compose(rows, f"{asset}  current corruption across states / views"),
                  f"{asset}__states__{tag}.png", sources,
                  {"kind": "state_grid", "asset": asset, "pairs": metrics})

    def teacher_view(self, asset, episode_dir, steps, references, episodes_root: Path):
        root = episodes_root / episode_dir
        episode = json.loads((root / "episode.json").read_text())
        by_step = {r["step"]: r for r in episode["trajectory"]}
        refs = [(f"reference shown to teacher\n{name.split('/')[-1]}", root / name) for name in references]
        obs = [(f"teacher observation step {s}\n{_action(by_step[s]['decision'])}",
                root / by_step[s]["image_ref"]) for s in steps]
        tiles = [[(label, load(path)) for label, path in refs], [(label, load(path)) for label, path in obs]]
        self.save(compose(tiles, f"{asset}  teacher inputs (saved files)"),
                  f"{asset}__teacher__{episode_dir}.png", [p for _, p in refs + obs],
                  {"kind": "teacher_view", "asset": asset, "episode": episode_dir, "steps": steps,
                   "references": references})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, default=ROOT / "output/interactive-teacher")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    builder = Builder(args.sweep.resolve(), args.out)
    for spec in KEY_PANELS:
        builder.key_panel(*spec)
    for spec in GRADIENTS:
        builder.gradient(*spec)
    for spec in STATE_GRIDS:
        builder.state_grid(*spec)
    for spec in TEACHER_VIEWS:
        builder.teacher_view(*spec, args.episodes_root.resolve())
    (args.out / "figures.json").write_text(json.dumps(builder.records, indent=1))


if __name__ == "__main__":
    main()
