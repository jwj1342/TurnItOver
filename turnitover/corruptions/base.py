"""Corruption contract and registry keyed by taxonomy defect id."""
from __future__ import annotations

from typing import Callable, ClassVar, Protocol

import numpy as np

from turnitover.core.program import ObjectProgram
from turnitover.core.sample import CorruptionSpec
from turnitover.taxonomy import load_taxonomy


class Corruption(Protocol):
    defect_id: ClassVar[str]

    def applicable(self, program: ObjectProgram) -> bool: ...

    def apply(self, program: ObjectProgram, rng: np.random.Generator) -> tuple[ObjectProgram, CorruptionSpec]: ...


_REGISTRY: dict[str, type] = {}


def register(defect_id: str) -> Callable[[type], type]:
    load_taxonomy().by_id(defect_id)  # raises for unknown ids

    def deco(cls: type) -> type:
        if defect_id in _REGISTRY:
            raise ValueError(f"corruption {defect_id} registered twice")
        cls.defect_id = defect_id
        _REGISTRY[defect_id] = cls
        return cls

    return deco


def get_corruption(defect_id: str, **params) -> Corruption:
    return _REGISTRY[defect_id](**params)


def registered_ids() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def require_spec(program: ObjectProgram):
    if program.spec is None:
        raise ValueError("spec-level corruption needs a program with a spec")
    return program.spec


def unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)
