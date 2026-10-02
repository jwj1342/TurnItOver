from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.runtime.actions import ViewAction


V1_NAME = "relative_discrete_v1"
V2_NAME = "pose_grid_v2"


@dataclass(frozen=True)
class CameraPose:
    view_id: str
    azimuth_deg: float
    elevation_deg: float
    distance_scale: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "view_id": self.view_id,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "distance_scale": self.distance_scale,
        }


@dataclass(frozen=True)
class CameraAction:
    """Action passed between the verifier and the observation runtime.

    M5 should depend on this representation rather than on OrbitControls mouse
    gestures or on a particular camera-pose implementation.
    """

    type: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "name": self.name, "params": dict(self.params)}


class CameraActionSpace(Protocol):
    name: str

    def resolve(self, action: str, *, view_id: str | None = None) -> CameraAction: ...

    def tool_instructions(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


class RelativeDiscreteActionSpace:
    """V1: discrete semantic relative actions backed by pointer gestures."""

    name = V1_NAME
    _allowed = {
        ViewAction.ORBIT_LEFT.value,
        ViewAction.ORBIT_RIGHT.value,
        ViewAction.ORBIT_UP.value,
        ViewAction.ORBIT_DOWN.value,
        ViewAction.ZOOM_IN.value,
        ViewAction.ZOOM_OUT.value,
        ViewAction.CAPTURE.value,
    }

    def __init__(self, *, orbit_drag_fraction: float = 0.22, zoom_wheel_delta: float = 700.0) -> None:
        self.orbit_drag_fraction = float(orbit_drag_fraction)
        self.zoom_wheel_delta = float(zoom_wheel_delta)

    def resolve(self, action: str, *, view_id: str | None = None) -> CameraAction:  # noqa: ARG002
        normalized = action.strip().lower()
        if normalized not in self._allowed:
            raise ValueError(
                f"action {action!r} is not available in {self.name}; "
                f"expected one of: {', '.join(sorted(self._allowed))}"
            )
        if normalized == ViewAction.CAPTURE.value:
            return CameraAction(type="capture", name=normalized)
        return CameraAction(
            type="relative",
            name=normalized,
            params={
                "orbit_drag_fraction": self.orbit_drag_fraction,
                "zoom_wheel_delta": self.zoom_wheel_delta,
            },
        )

    def tool_instructions(self) -> str:
        return (
            "当前相机动作空间为 V1 relative_discrete_v1。允许动作："
            "orbit_left、orbit_right、orbit_up、orbit_down、zoom_in、zoom_out、capture。"
            "这些动作表示相对于当前视角的离散语义移动；V1 不承诺精确角度。"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "orbit_drag_fraction": self.orbit_drag_fraction,
            "zoom_wheel_delta": self.zoom_wheel_delta,
            "actions": sorted(self._allowed),
        }


class PoseGridActionSpace:
    """V2: finite set of explicit canonical spherical camera poses."""

    name = V2_NAME

    def __init__(
        self,
        poses: list[CameraPose],
        *,
        initial_view_id: str,
        harness_name: str = "__img2threejsCameraV2",
    ) -> None:
        if not poses:
            raise ValueError("pose_grid_v2 requires at least one camera pose")
        self.poses = {pose.view_id: pose for pose in poses}
        if len(self.poses) != len(poses):
            raise ValueError("pose_grid_v2 contains duplicate view ids")
        if initial_view_id not in self.poses:
            raise ValueError(f"unknown V2 initial_view_id: {initial_view_id}")
        self.initial_view_id = initial_view_id
        self.harness_name = harness_name

    def resolve(self, action: str, *, view_id: str | None = None) -> CameraAction:
        normalized = action.strip().lower()
        if normalized == "capture":
            return CameraAction(type="capture", name="capture")
        if normalized != "goto_pose":
            raise ValueError(
                f"action {action!r} is not available in {self.name}; expected goto_pose or capture"
            )
        if not view_id:
            raise ValueError("goto_pose requires view_id")
        try:
            pose = self.poses[view_id]
        except KeyError as exc:
            raise ValueError(f"unknown V2 pose view_id: {view_id}") from exc
        return CameraAction(type="goto_pose", name="goto_pose", params=pose.to_dict())

    def initial_action(self) -> CameraAction:
        return self.resolve("goto_pose", view_id=self.initial_view_id)

    def tool_instructions(self) -> str:
        ids = ", ".join(self.poses)
        return (
            "当前相机动作空间为 V2 pose_grid_v2。使用 action='goto_pose' 并提供 view_id，"
            f"可选 view_id：{ids}。也可以使用 action='capture'。"
            "每个 view_id 对应确定的 canonical azimuth/elevation/distance pose；不要输出任意浮点角度。"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial_view_id": self.initial_view_id,
            "harness_name": self.harness_name,
            "poses": [pose.to_dict() for pose in self.poses.values()],
        }


def load_action_space(mode: str, config_path: str | Path | None = None) -> CameraActionSpace:
    normalized = mode.strip().lower()
    data: dict[str, Any] = {}
    if config_path:
        path = Path(config_path)
        data = json.loads(path.read_text(encoding="utf-8"))

    if normalized == V1_NAME:
        relative = data.get("relative", data)
        return RelativeDiscreteActionSpace(
            orbit_drag_fraction=float(relative.get("orbit_drag_fraction", 0.22)),
            zoom_wheel_delta=float(relative.get("zoom_wheel_delta", 700.0)),
        )

    if normalized == V2_NAME:
        pose_items = data.get("poses") or []
        poses = [
            CameraPose(
                view_id=str(item["view_id"]),
                azimuth_deg=float(item["azimuth_deg"]),
                elevation_deg=float(item["elevation_deg"]),
                distance_scale=float(item["distance_scale"]),
            )
            for item in pose_items
        ]
        return PoseGridActionSpace(
            poses,
            initial_view_id=str(data.get("initial_view_id", "")),
            harness_name=str(data.get("harness_name", "__img2threejsCameraV2")),
        )

    raise ValueError(f"unknown VIEW_ACTION_SPACE {mode!r}; expected {V1_NAME} or {V2_NAME}")
