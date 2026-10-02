from __future__ import annotations

from enum import Enum

from src.runtime.actions import ViewAction


class VerifierAction(str, Enum):
    ORBIT_LEFT = "orbit_left"
    ORBIT_RIGHT = "orbit_right"
    ORBIT_UP = "orbit_up"
    ORBIT_DOWN = "orbit_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    CAPTURE = "capture"
    FINISH = "finish"

    def as_view_action(self) -> ViewAction:
        if self is VerifierAction.FINISH:
            raise ValueError("finish is not a browser view action")
        return ViewAction(self.value)


def parse_verifier_action(value: str) -> VerifierAction:
    try:
        return VerifierAction(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(action.value for action in VerifierAction)
        raise ValueError(f"unknown verifier action {value!r}; expected one of: {valid}") from exc
