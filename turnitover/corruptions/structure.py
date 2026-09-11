from __future__ import annotations

import numpy as np

from turnitover.core.program import ObjectProgram
from turnitover.core.sample import CorruptionSpec
from turnitover.corruptions.base import register, require_spec, unit_vector


@register("structure.part_offset")
class PartOffset:
    """Translate one part by a random vector. Static-visible."""

    def __init__(self, min_m: float = 0.02, max_m: float = 0.08):
        self.min_m, self.max_m = min_m, max_m

    def applicable(self, program: ObjectProgram) -> bool:
        return program.spec is not None and len(program.spec.parts) > 0

    def apply(self, program: ObjectProgram, rng: np.random.Generator):
        spec = require_spec(program)
        part = spec.parts[rng.integers(len(spec.parts))]
        mag = float(rng.uniform(self.min_m, self.max_m))
        vec = unit_vector(rng) * mag
        origin = tuple(float(o + d) for o, d in zip(part.origin, vec))
        new_spec = spec.replace_part(part.id, origin=origin)
        cs = CorruptionSpec(
            defect_id=self.defect_id, parts=(part.id,), severity=mag / self.max_m,
            params={"magnitude_m": mag, "vector": vec.tolist()},
            label={"part": part.id, "vector": vec.tolist()},
        )
        return ObjectProgram.from_spec(new_spec), cs
