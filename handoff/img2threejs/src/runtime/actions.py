from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ViewAction(str, Enum):
    """Small, model-friendly action space for interacting with a rendered canvas."""

    ORBIT_LEFT = "orbit_left"
    ORBIT_RIGHT = "orbit_right"
    ORBIT_UP = "orbit_up"
    ORBIT_DOWN = "orbit_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    CAPTURE = "capture"


@dataclass(frozen=True)
class PointerMotion:
    dx: float = 0.0
    dy: float = 0.0
    wheel_y: float = 0.0


_DRAG_FRACTION = 0.22
_WHEEL_DELTA = 700.0


def pointer_motion(action: ViewAction, *, width: float, height: float) -> PointerMotion:
    """Translate a verifier action into a browser pointer gesture."""
    dx = max(width * _DRAG_FRACTION, 48.0)
    dy = max(height * _DRAG_FRACTION, 48.0)

    if action == ViewAction.ORBIT_LEFT:
        return PointerMotion(dx=-dx)
    if action == ViewAction.ORBIT_RIGHT:
        return PointerMotion(dx=dx)
    if action == ViewAction.ORBIT_UP:
        return PointerMotion(dy=-dy)
    if action == ViewAction.ORBIT_DOWN:
        return PointerMotion(dy=dy)
    if action == ViewAction.ZOOM_IN:
        return PointerMotion(wheel_y=-_WHEEL_DELTA)
    if action == ViewAction.ZOOM_OUT:
        return PointerMotion(wheel_y=_WHEEL_DELTA)
    return PointerMotion()


def parse_action(value: str) -> ViewAction:
    try:
        return ViewAction(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(action.value for action in ViewAction)
        raise ValueError(f"unknown view action {value!r}; expected one of: {valid}") from exc
