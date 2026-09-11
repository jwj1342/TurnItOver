"""DetectabilityMatrix computed empirically on an asset.

Definition (perfect-observer proxy): run the same single action on a corrupted program and
on the reference program; the action *detects* the corruption when the returned evidence
differs beyond a tolerance. Views compare pixels; joint actuation compares exported geometry;
runtime stats compare numbers.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from turnitover.checkers import meshops
from turnitover.core.actions import Action, ActuateJoint, QueryRuntime, RequestView, action_key
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing
from turnitover.corruptions import Corruption
from turnitover.engine.sharding import sample_seed
from turnitover.render.session import ObservationSession


@dataclass(frozen=True)
class DetectabilityEntry:
    detect_rate: float
    n: int
    mean_signal: float


@dataclass(frozen=True)
class Thresholds:
    image_l1: float = 0.01  # mean |Δ| over normalized RGB
    geometry_chamfer_m: float = 0.005
    stats_rel: float = 0.05


@dataclass
class DetectabilityMatrix:
    entries: dict[tuple[str, str], DetectabilityEntry]

    def to_json(self, path: Path) -> None:
        rows = [{"defect_id": d, "action_key": a, **e.__dict__} for (d, a), e in self.entries.items()]
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "DetectabilityMatrix":
        rows = json.loads(path.read_text(encoding="utf-8"))
        return DetectabilityMatrix({(r["defect_id"], r["action_key"]): DetectabilityEntry(r["detect_rate"], r["n"], r["mean_signal"]) for r in rows})

    def to_markdown(self) -> str:
        defects = sorted({d for d, _ in self.entries})
        actions = sorted({a for _, a in self.entries})
        lines = ["| defect \\ action | " + " | ".join(actions) + " |", "|---|" + "---|" * len(actions)]
        for d in defects:
            cells = [f"{self.entries[(d, a)].detect_rate:.2f}" if (d, a) in self.entries else "" for a in actions]
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


def _observe(session: ObservationSession, program: ObjectProgram, framing: Framing, action: Action):
    session.load(program, framing=framing)
    try:
        match action:
            case RequestView(view_id=v):
                return _img(session.request_view(v).png)
            case ActuateJoint(joint_id=j, detent=d):
                session.actuate(j, d)
                return session.export_geometry()
            case QueryRuntime(property=p):
                return session.query_runtime(p)
        raise TypeError(action)
    finally:
        session.dispose()


def _img(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255.0


def _signal(action: Action, a, b, thr: Thresholds) -> tuple[float, bool]:
    match action:
        case RequestView():
            s = float(np.abs(a - b).mean())
            return s, s > thr.image_l1
        case ActuateJoint():
            worst = 0.0
            for pid in set(a) | set(b):
                if pid not in a or pid not in b:
                    return 1.0, True
                worst = max(worst, meshops.surface_chamfer(a[pid].vertices, a[pid].faces, b[pid].vertices, b[pid].faces, 1000))
            return worst, worst > thr.geometry_chamfer_m
        case QueryRuntime():
            keys = [k for k, v in a.items() if isinstance(v, (int, float)) and k != "last_render_ms"]
            rel = max((abs(float(a[k]) - float(b.get(k, 0))) / max(abs(float(a[k])), 1.0) for k in keys), default=0.0)
            return rel, rel > thr.stats_rel
    raise TypeError(action)


def compute_on_asset(
    session: ObservationSession,
    reference: ObjectProgram,
    corruptions: Sequence[Corruption],
    actions: Sequence[Action],
    n_seeds: int,
    base_seed: int = 0,
    thresholds: Thresholds = Thresholds(),
) -> DetectabilityMatrix:
    info = session.load(reference, framing=None)
    framing = info.framing
    session.dispose()
    ref_obs = {action_key(a): _observe(session, reference, framing, a) for a in actions}
    entries: dict[tuple[str, str], DetectabilityEntry] = {}
    for corr in corruptions:
        hits: dict[str, list[tuple[bool, float]]] = {action_key(a): [] for a in actions}
        for s in range(n_seeds):
            rng = np.random.default_rng(sample_seed(base_seed, s))
            program, _ = corr.apply(reference, rng)
            for a in actions:
                obs = _observe(session, program, framing, a)
                sig, hit = _signal(a, obs, ref_obs[action_key(a)], thresholds)
                hits[action_key(a)].append((hit, sig))
        for key, rows in hits.items():
            entries[(corr.defect_id, key)] = DetectabilityEntry(
                detect_rate=float(np.mean([h for h, _ in rows])), n=len(rows), mean_signal=float(np.mean([s for _, s in rows]))
            )
    return DetectabilityMatrix(entries)
