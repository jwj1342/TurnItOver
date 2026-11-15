"""Corruption contract and shared spec operations."""
from __future__ import annotations

from typing import ClassVar, Protocol

import numpy as np

from turnitover.core.program import ObjectProgram
from turnitover.core.sample import CorruptionSpec


class Corruption(Protocol):
    defect_id: ClassVar[str]

    def applicable(self, program: ObjectProgram) -> bool: ...

    def apply(self, program: ObjectProgram, rng: np.random.Generator) -> tuple[ObjectProgram, CorruptionSpec]: ...


def require_spec(program: ObjectProgram):
    if program.spec is None:
        raise ValueError("spec-level corruption needs a program with a spec")
    return program.spec


def unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)
