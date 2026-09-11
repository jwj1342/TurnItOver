"""The judge's action space and the observation record it produces.

``action_key`` is the column key of the detectability matrix; keep it stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

ViewId = NewType("ViewId", str)


class Detent(StrEnum):
    ZERO = "zero"
    MID = "mid"
    LIMIT = "limit"


class RuntimeProperty(StrEnum):
    STATS = "stats"
    HIERARCHY = "hierarchy"
    JOINT_STATE = "joint_state"
    STATE_DELTA = "state_delta"


@dataclass(frozen=True)
class RequestView:
    view_id: ViewId


@dataclass(frozen=True)
class ActuateJoint:
    joint_id: str
    detent: Detent


@dataclass(frozen=True)
class QueryRuntime:
    property: RuntimeProperty


@dataclass(frozen=True)
class EmitDiagnosis:
    parts: tuple[str, ...]
    defect_ids: tuple[str, ...]
    severity: float
    text: str


@dataclass(frozen=True)
class Stop:
    pass


Action = RequestView | ActuateJoint | QueryRuntime | EmitDiagnosis | Stop
OBSERVATION_ACTIONS = (RequestView, ActuateJoint, QueryRuntime)


def action_key(action: Action) -> str:
    match action:
        case RequestView(view_id=v):
            return f"view:{v}"
        case ActuateJoint(joint_id=j, detent=d):
            return f"joint:{j}:{d.value}"
        case QueryRuntime(property=p):
            return f"runtime:{p.value}"
        case EmitDiagnosis():
            return "emit"
        case Stop():
            return "stop"
    raise TypeError(f"unknown action {action!r}")


def action_cost(action: Action) -> int:
    return 1 if isinstance(action, OBSERVATION_ACTIONS) else 0


def detent_value(detent: Detent, limits: tuple[float, float]) -> float:
    lo, hi = limits
    match detent:
        case Detent.ZERO:
            return lo
        case Detent.MID:
            return 0.5 * (lo + hi)
        case Detent.LIMIT:
            return hi
    raise ValueError(detent)


def joint_state_key(joint_id: str, detent: Detent) -> str:
    """State key used by evidence/checkers for geometry captured at a detent."""
    return f"joint:{joint_id}:{detent.value}"


REST_STATE = "rest"


@dataclass(frozen=True)
class Observation:
    step: int
    action: Action
    image_ref: str | None
    payload: dict | None
    elapsed_ms: float
