"""Python mirror of web/src/protocol.ts plus decoding helpers."""
from __future__ import annotations

import base64
from typing import Literal, TypedDict

import numpy as np

Vec3 = tuple[float, float, float]


class ViewDefDict(TypedDict, total=False):
    id: str
    azimuth_deg: float
    elevation_deg: float
    projection: Literal["perspective", "orthographic"]
    distance_factor: float


class FramingDict(TypedDict):
    center: list[float]
    radius: float


class LoadResponse(TypedDict):
    ok: bool
    parts: list[str]
    joints: list[dict]
    framing: FramingDict


class ImageResponse(TypedDict):
    image_png_b64: str
    width: int
    height: int
    render_ms: float


class StatsPayload(TypedDict):
    property: str
    triangles_rendered: int
    draw_calls: int
    triangles_scene: int
    mesh_count: int
    geometries: int
    textures: int
    programs: int
    last_render_ms: float


class PartGeometryDict(TypedDict):
    name: str
    vertex_count: int
    triangle_count: int
    positions_b64: str
    indices_b64: str


class GeometryExportDict(TypedDict):
    joint_state: dict[str, float]
    parts: list[PartGeometryDict]


STATS_KEYS = tuple(StatsPayload.__annotations__)


def decode_png(b64: str) -> bytes:
    return base64.b64decode(b64)


def decode_positions(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="<f4").reshape(-1, 3)


def decode_indices(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="<u4").reshape(-1, 3)
