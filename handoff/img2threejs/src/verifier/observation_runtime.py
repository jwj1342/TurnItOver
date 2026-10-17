from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.runtime import BrowserSession
from src.runtime.actions import ViewAction, pointer_motion

from .action_space import CameraAction, PoseGridActionSpace, RelativeDiscreteActionSpace


class ObservationRuntime(Protocol):
    """Backend that turns semantic camera actions into rendered observations."""

    action_space_name: str

    def reset(self, html: str) -> None: ...

    def prepare_loaded(self) -> None: ...

    def execute(self, action: CameraAction, screenshot_path: str | Path) -> dict[str, Any]: ...


@dataclass
class RelativeBrowserRuntime:
    """V1 runtime: execute relative semantic actions as browser pointer gestures."""

    session: BrowserSession
    action_space: RelativeDiscreteActionSpace

    @property
    def action_space_name(self) -> str:
        return self.action_space.name

    def reset(self, html: str) -> None:
        load = self.session.load_html(html)
        if not load.success:
            errors = load.load_errors + load.page_errors + load.console_errors
            raise RuntimeError(
                "RelativeBrowserRuntime requires renderable HTML; load failed with: "
                + " | ".join(errors)
            )

    def prepare_loaded(self) -> None:
        # The pipeline render gate has already loaded and validated this exact
        # candidate. Keep that document instead of fetching its dependencies a
        # second time immediately before verification.
        self.session._largest_canvas()  # intentional runtime seam

    def execute(self, action: CameraAction, screenshot_path: str | Path) -> dict[str, Any]:
        if action.type == "capture":
            self.session.capture(screenshot_path)
            return {"action": action.to_dict(), "backend": "capture"}
        if action.type != "relative":
            raise ValueError(f"V1 runtime cannot execute camera action type {action.type!r}")

        view_action = ViewAction(action.name)
        # Keep V1 parameters configurable even though BrowserSession.perform has
        # historical defaults. Execute the gesture here using the requested values.
        locator, box = self.session._largest_canvas()  # intentional runtime seam
        page = self.session._page
        assert page is not None
        center_x = box["x"] + box["width"] / 2.0
        center_y = box["y"] + box["height"] / 2.0
        drag_fraction = float(action.params.get("orbit_drag_fraction", 0.22))
        wheel_delta = float(action.params.get("zoom_wheel_delta", 700.0))
        dx = max(box["width"] * drag_fraction, 48.0)
        dy = max(box["height"] * drag_fraction, 48.0)

        if view_action is ViewAction.ZOOM_IN:
            locator.hover()
            page.mouse.wheel(0, -wheel_delta)
        elif view_action is ViewAction.ZOOM_OUT:
            locator.hover()
            page.mouse.wheel(0, wheel_delta)
        else:
            move_x = 0.0
            move_y = 0.0
            if view_action is ViewAction.ORBIT_LEFT:
                move_x = -dx
            elif view_action is ViewAction.ORBIT_RIGHT:
                move_x = dx
            elif view_action is ViewAction.ORBIT_UP:
                move_y = -dy
            elif view_action is ViewAction.ORBIT_DOWN:
                move_y = dy
            page.mouse.move(center_x, center_y)
            page.mouse.down()
            page.mouse.move(center_x + move_x, center_y + move_y, steps=8)
            page.mouse.up()

        page.wait_for_timeout(self.session.config.settle_ms)
        self.session.capture(screenshot_path)
        return {"action": action.to_dict(), "backend": "pointer_gesture"}


@dataclass
class PoseBrowserRuntime:
    """V2 runtime: execute deterministic absolute poses through a page harness.

    The page must expose ``window[<harness_name>]`` with synchronous methods:
      - setPose({azimuth_deg, elevation_deg, distance_scale})
      - getPose() -> object
    The runtime never falls back to pointer gestures, because silent fallback
    would destroy cross-program determinism.
    """

    session: BrowserSession
    action_space: PoseGridActionSpace

    @property
    def action_space_name(self) -> str:
        return self.action_space.name

    def reset(self, html: str) -> None:
        load = self.session.load_html(html)
        if not load.success:
            errors = load.load_errors + load.page_errors + load.console_errors
            raise RuntimeError(
                "PoseBrowserRuntime requires renderable HTML; load failed with: "
                + " | ".join(errors)
            )
        self.prepare_loaded()

    def prepare_loaded(self) -> None:
        self._ensure_harness()
        self.execute(self.action_space.initial_action(), Path("/dev/null"), capture=False)

    def execute(
        self,
        action: CameraAction,
        screenshot_path: str | Path,
        *,
        capture: bool = True,
    ) -> dict[str, Any]:
        if action.type == "capture":
            if capture:
                self.session.capture(screenshot_path)
            return {"action": action.to_dict(), "backend": "capture", "pose": self._get_pose()}
        if action.type != "goto_pose":
            raise ValueError(f"V2 runtime cannot execute camera action type {action.type!r}")

        self._ensure_harness()
        params = {
            "azimuth_deg": float(action.params["azimuth_deg"]),
            "elevation_deg": float(action.params["elevation_deg"]),
            "distance_scale": float(action.params["distance_scale"]),
        }
        page = self.session._page
        assert page is not None
        page.evaluate(
            "([name, pose]) => window[name].setPose(pose)",
            [self.action_space.harness_name, params],
        )
        page.wait_for_timeout(self.session.config.settle_ms)
        pose = self._get_pose()
        if capture:
            self.session.capture(screenshot_path)
        return {"action": action.to_dict(), "backend": "camera_harness", "pose": pose}

    def _ensure_harness(self) -> None:
        page = self.session._page
        assert page is not None
        ok = page.evaluate(
            "name => { const h = window[name]; return !!h && typeof h.setPose === 'function' && typeof h.getPose === 'function'; }",
            self.action_space.harness_name,
        )
        if not ok:
            raise RuntimeError(
                "pose_grid_v2 requires deterministic camera harness "
                f"window.{self.action_space.harness_name}.setPose/getPose"
            )

    def _get_pose(self) -> dict[str, Any]:
        page = self.session._page
        assert page is not None
        result = page.evaluate(
            "name => window[name].getPose()",
            self.action_space.harness_name,
        )
        return dict(result or {})
