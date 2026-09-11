"""GeometryAlignmentChecker: per-state, per-part bbox IoU and Chamfer vs the reference program."""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from turnitover.checkers import meshops
from turnitover.checkers.base import Evidence
from turnitover.core.sample import CheckResult, Invariant


class GeometryAlignmentChecker:
    checker_id: ClassVar[str] = "geometry.alignment"
    needs_reference: ClassVar[bool] = True

    def __init__(self, tol_chamfer_m: float = 0.005, tol_iou: float = 0.9, samples_per_part: int = 2000, seed: int = 0):
        self.tol_chamfer_m = tol_chamfer_m
        self.tol_iou = tol_iou
        self.samples_per_part = samples_per_part
        self.seed = seed

    def check(self, evidence: Evidence, reference: Evidence | None) -> CheckResult:
        assert reference is not None, "geometry.alignment needs a reference"
        inv: list[Invariant] = []
        if not evidence.compile_ok:
            inv.append(Invariant("compile", 0.0, 0.0, False))
            return CheckResult(self.checker_id, False, tuple(inv), 0.0)
        for state, ref_parts in reference.geometry.items():
            cand_parts = evidence.geometry.get(state, {})
            for pid in sorted(set(ref_parts) - set(cand_parts)):
                inv.append(Invariant("part.missing", 1.0, 0.0, False, (pid,), state))
            for pid in sorted(set(cand_parts) - set(ref_parts)):
                inv.append(Invariant("part.extra", 1.0, 0.0, False, (pid,), state))
            for pid in sorted(set(ref_parts) & set(cand_parts)):
                r, c = ref_parts[pid], cand_parts[pid]
                iou = meshops.aabb_iou(r.vertices, c.vertices)
                ch = meshops.surface_chamfer(r.vertices, r.faces, c.vertices, c.faces, self.samples_per_part, self.seed)
                inv.append(Invariant("bbox_iou", iou, self.tol_iou, iou >= self.tol_iou, (pid,), state))
                inv.append(Invariant("chamfer_m", ch, self.tol_chamfer_m, ch <= self.tol_chamfer_m, (pid,), state))
        passed = all(i.passed for i in inv)
        score = float(np.mean([i.passed for i in inv])) if inv else 1.0
        return CheckResult(self.checker_id, passed, tuple(inv), score)
