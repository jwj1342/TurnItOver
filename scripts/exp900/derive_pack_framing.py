"""Recover the fixed camera framing the three-tier-900 release builders used for every case of her repair pack.

The pack stores no framing. Recipe, verified byte-for-byte against the pack's frozen images
(scripts/exp900/check_pack_framing.py): load the reference program, take the exact world-space vertex AABB at the
target joint's rest, mid and limit state values (other joints at zero) and unite them; center = box center,
radius = half the box diagonal times a per-tier margin. The margins differ because Easy, Medium and Hard were
rendered by three different release builders. Reference and candidate share this framing.

Loads each reference once and reads its geometry; renders nothing. Writes turnitover/repair/framing_900.json,
which load_pack reads.

Usage: python scripts/exp900/derive_pack_framing.py --pack $TIO_RUNS/data900/repair58/three-tier-900-validation-repair-58-v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.repair.judges import FRAMING_900

MARGIN = {"easy": 1.25, "medium": 1.5, "hard": 1.15}


def swept_box(session: ObservationSession, joint: str, values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []
    for value in values:
        session.reset_joints()
        session.set_joints({joint: value})
        vertices = np.concatenate([mesh.vertices for mesh in session.export_geometry().values()]).astype(float)
        lows.append(vertices.min(0))
        highs.append(vertices.max(0))
    return np.min(lows, 0), np.max(highs, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/repair_pack900.yaml"))
    parser.add_argument("--write", type=Path, default=FRAMING_900)
    args = parser.parse_args()
    cfg = load_generate_config(args.config)
    cases = {}
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
        for entry in json.loads((args.pack / "cases.json").read_text()):
            session.load(ObjectProgram((args.pack / entry["programs"]["reference"]["typescript"]).read_text()), framing=None)
            low, high = swept_box(session, entry["target_joint_id"], [float(v) for v in entry["state_values"].values()])
            session.dispose()
            radius = float(np.linalg.norm(high - low)) / 2 * MARGIN[entry["difficulty"]]
            cases[entry["case_id"]] = {"center": [float(x) for x in (low + high) / 2], "radius": radius}
            print(json.dumps({"case": entry["case_id"], **cases[entry["case_id"]]}), flush=True)
    table = {"derived_by": "scripts/exp900/derive_pack_framing.py", "pack": json.loads((args.pack / "manifest.json").read_text())["format"],
             "margin": MARGIN, "cases": dict(sorted(cases.items()))}
    args.write.write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
