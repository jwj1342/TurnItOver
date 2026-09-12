from __future__ import annotations

import math

import numpy as np

from turnitover.core.program import ObjectProgram
from turnitover.core.sample import CorruptionSpec
from turnitover.corruptions.base import register, require_spec


@register("runtime.triangle_budget")
class TriangleBudget:
    """Multiply box segments of k parts so the triangle count explodes. Probe-only visible."""

    FACTORS = (8, 16, 32)

    def __init__(self, k_parts: int = 2):
        self.k_parts = k_parts

    def applicable(self, program: ObjectProgram) -> bool:
        return program.spec is not None and any(p.mesh is None for p in program.spec.parts)

    def apply(self, program: ObjectProgram, rng: np.random.Generator):
        spec = require_spec(program)
        boxes = [p for p in spec.parts if p.mesh is None]
        if not boxes:
            raise ValueError("Triangle subdivision corruption currently requires box parts")
        k = min(self.k_parts, len(boxes))
        chosen = [boxes[i] for i in rng.choice(len(boxes), size=k, replace=False)]
        factor = int(self.FACTORS[rng.integers(len(self.FACTORS))])
        new_spec = spec
        for p in chosen:
            new_spec = new_spec.replace_part(p.id, segments=tuple(int(s * factor) for s in p.segments))
        before = _box_triangles(spec)
        after = _box_triangles(new_spec)
        ratio = after / max(before, 1)
        max_ratio = 1 + (self.FACTORS[-1] ** 2 - 1)  # one part fully scaled dominates
        cs = CorruptionSpec(
            defect_id=self.defect_id, parts=tuple(p.id for p in chosen),
            severity=min(1.0, math.log(ratio) / math.log(max_ratio)),
            params={"factor": factor, "triangles_before": before, "triangles_after": after},
            label={"scalar": after},
        )
        return ObjectProgram.from_spec(new_spec), cs


def _box_triangles(spec) -> int:
    total = 0
    for p in spec.parts:
        if p.mesh is not None:
            total += len(p.mesh.faces)
            continue
        sx, sy, sz = p.segments
        total += 2 * 2 * (sx * sy + sy * sz + sx * sz)
    return total
