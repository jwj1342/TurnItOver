"""Render the round-0 sweep of repair-pack cases exactly as the repair loop does (load_pack framing, sweep_cells) and
compare every reference and candidate image with her frozen pack image of the same name.

Prints one TSV row per image: identical bytes, number of differing pixels, max and mean absolute RGBA difference, and
the object's pixel bounding box (x0, y0, x1, y1; pixels that differ from the background) in the pack image and in ours.
Login node: at most a handful of cases; all 58 belong in a CPU job.

Usage: python scripts/exp900/check_pack_framing.py --pack $TIO_RUNS/data900/repair58/three-tier-900-validation-repair-58-v1 --case-ids eb-0f542119f7b0 ...
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from turnitover.config import load_generate_config
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.repair.judges import load_pack, sweep_cells

BACKGROUND = np.array([32, 32, 36])     # harness scene background 0x202024


def pixels(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA")).astype(int)


def object_box(image: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.abs(image[..., :3] - BACKGROUND).sum(-1) > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/repair_pack900.yaml"))
    args = parser.parse_args()
    cfg = load_generate_config(args.config)
    cases = [c for c in load_pack(args.pack) if c.uid in set(args.case_ids)]
    print("case\ttier\timage\tidentical\tdiffering_pixels\tmax_abs\tmean_abs\tbox_pack\tbox_ours")
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session, tempfile.TemporaryDirectory() as cache:
        for case in cases:
            cells, _ = sweep_cells(session, case, case.initial, Path(cache))
            for cell in cells:
                pack = args.pack / "images" / case.uid / cell.name
                ours, theirs = pixels(cell.path), pixels(pack)
                difference = np.abs(ours - theirs)
                print(f"{case.uid}\t{case.tier}\t{cell.name}\t{cell.path.read_bytes() == pack.read_bytes()}\t"
                      f"{int((difference.max(-1) > 0).sum())}\t{difference.max()}\t{difference.mean():.4f}\t"
                      f"{object_box(theirs)}\t{object_box(ours)}", flush=True)


if __name__ == "__main__":
    main()
