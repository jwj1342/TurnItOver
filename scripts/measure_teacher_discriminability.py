"""Auditor-side measurement of how discriminable a motion defect is in each frozen witness view.

For every motion-view witness state the candidate is compared with the reference at the
same state (defect signal), the part of that change at least 5 pixels thick (core signal), and the reference state with the reference rest pose (motion
signal: how much the visible articulation itself moves). Oracle exposure only says that
some pixels differ; the ratio of the two signals and the side-by-side panels written here
are used to judge whether that difference is plausibly perceptible. Nothing produced here
is ever shown to the teacher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.oracle.evidence import ExposureThresholds, pixel_signal, pixels
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.teacher.witnesses import _render, core_changed_fraction, file_sha256, sample_framing


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/benchmarks/partnet-mobility-formal-v1/prepared"
DEFAULT_CONFIG = ROOT / "configs/generate_partnet_mobility_supervision.yaml"


def _rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}


def _changed_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.abs(a.astype(np.float32) - b.astype(np.float32)).max(axis=2) / 255.
            > ExposureThresholds().pixel_delta)


def _changed_box(mask: np.ndarray) -> list[int] | None:
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _panel(reference: np.ndarray, candidate: np.ndarray, path: Path) -> None:
    """Reference | candidate | 4x amplified absolute difference."""
    delta = np.clip(np.abs(reference.astype(np.int16) - candidate.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    Image.fromarray(np.concatenate([reference, candidate, delta], axis=1)).save(path)


def run(args) -> list[dict]:
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = load_generate_config(args.config)
    gold = _rows(args.benchmark / "train.gold.jsonl")
    results = []
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
        for uid in args.sample_id:
            manifest = json.loads((args.frozen_witnesses / uid / "manifest.json").read_text())
            row = gold[uid]
            framing = sample_framing(args.benchmark.parent / "raw", row["source_id"], row["sample_id"])
            candidate, reference = ObjectProgram(row["candidate_program"]), ObjectProgram(row["reference_program"])
            rest = {}
            states = []
            for witness in manifest["witnesses"]:
                cand_png = _render(session, candidate, framing, witness["view"], witness["state"])
                if hashlib.sha256(cand_png).hexdigest() != witness["sha256"]:
                    raise ValueError(f"Candidate render differs from frozen witness {uid}/{witness['index']}")
                ref = pixels(_render(session, reference, framing, witness["view"], witness["state"]))
                cand = pixels(cand_png)
                if witness["joint"] is None:
                    rest[witness["view"]] = ref
                defect = pixel_signal(cand, ref)
                mask = _changed_mask(cand, ref)
                motion = pixel_signal(ref, rest[witness["view"]])
                panel = args.out / uid / f"{witness['key'].replace(':', '_')}.png"
                panel.parent.mkdir(parents=True, exist_ok=True)
                _panel(ref, cand, panel)
                states.append({
                    "index": witness["index"], "key": witness["key"], "view": witness["view"],
                    "exposures": witness["exposures"],
                    "defect_changed_fraction": defect["changed_fraction"], "defect_image_l1": defect["image_l1"],
                    "reference_motion_changed_fraction": motion["changed_fraction"],
                    "defect_to_motion_ratio": (defect["changed_fraction"] / motion["changed_fraction"]
                                               if motion["changed_fraction"] else None),
                    "defect_core_changed_fraction": core_changed_fraction(cand, ref),
                    "defect_box_xyxy": _changed_box(mask), "panel": str(panel.relative_to(args.out)),
                    "panel_sha256": file_sha256(panel)})
            results.append({"id": uid, "asset_id": manifest["asset_id"],
                            "candidate_joint_limits": manifest["candidate_joint_limits"], "states": states})
            print(json.dumps({"id": uid, "states": len(states)}), flush=True)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--frozen-witnesses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run(parser.parse_args())
