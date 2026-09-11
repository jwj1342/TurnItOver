"""Checker contract. Checkers see only Evidence (what the browser exported), never AssetSpec."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping, Protocol

import numpy as np

from turnitover.core.sample import CheckResult


@dataclass(frozen=True)
class PartMesh:
    vertices: np.ndarray  # (N, 3) float32, world space
    faces: np.ndarray  # (M, 3) uint32


@dataclass(frozen=True)
class Evidence:
    program_sha: str
    compile_ok: bool
    error: str | None
    geometry: Mapping[str, Mapping[str, PartMesh]]  # state_key -> part_id -> mesh
    stats: dict
    hierarchy: dict


class Checker(Protocol):
    checker_id: ClassVar[str]
    needs_reference: ClassVar[bool]

    def check(self, evidence: Evidence, reference: Evidence | None) -> CheckResult: ...
