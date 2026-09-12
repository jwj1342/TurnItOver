"""Sample: one fully labeled record written to disk by the data engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from turnitover.core.actions import Observation

SCHEMA_VERSION = 3


@dataclass(frozen=True)
class Framing:
    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class CorruptionSpec:
    defect_id: str
    parts: tuple[str, ...]
    severity: float
    params: dict[str, Any] = field(default_factory=dict)
    label: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Invariant:
    name: str
    value: float
    tolerance: float
    passed: bool
    parts: tuple[str, ...] = ()
    state: str | None = None


@dataclass(frozen=True)
class CheckResult:
    checker_id: str
    passed: bool
    invariants: tuple[Invariant, ...]
    score: float | None = None


@dataclass(frozen=True)
class Sample:
    schema_version: int
    sample_id: str
    run_id: str
    idx: int
    seed: int
    asset_id: str
    source_id: str
    program_sha: str
    program_source: str
    reference_sha: str
    corruptions: tuple[CorruptionSpec, ...]
    framing: Framing | None
    trajectory: tuple[Observation, ...]
    checks: tuple[CheckResult, ...]
    compile_ok: bool
    error: str | None
    timing_ms: dict[str, float]
    host: str
    created_at: str
    dataset: dict[str, Any] = field(default_factory=dict)
