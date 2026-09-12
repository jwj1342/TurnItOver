"""AssetSpec: the canonical, mutable-by-replacement description of an articulated object.

Corruptions operate on this; the Three.js program is derived from it by
``turnitover.assets.emit``. Parts support boxes or embedded indexed triangle meshes.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum

Vec3 = tuple[float, float, float]
IVec3 = tuple[int, int, int]


class JointType(StrEnum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


@dataclass(frozen=True)
class MaterialSpec:
    id: str
    color: str = "#c8c8c8"
    metalness: float = 0.0
    roughness: float = 0.8


@dataclass(frozen=True)
class MeshSpec:
    vertices: tuple[Vec3, ...]
    faces: tuple[IVec3, ...]
    colors: tuple[Vec3, ...] = ()  # linear RGB, optionally one per vertex


@dataclass(frozen=True)
class PartSpec:
    id: str
    parent: str | None
    origin: Vec3
    rotation: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3 = (0.1, 0.1, 0.1)
    segments: IVec3 = (1, 1, 1)
    material: str = "default"
    mesh: MeshSpec | None = None  # None = box; empty mesh = transform-only link


@dataclass(frozen=True)
class JointSpec:
    id: str
    part: str
    type: JointType
    axis: Vec3
    anchor: Vec3
    limits: tuple[float, float]
    sign: float = 1.0
    frame_rotation: Vec3 = (0.0, 0.0, 0.0)  # Three.js intrinsic XYZ, before joint motion


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    category: str
    parts: tuple[PartSpec, ...]
    joints: tuple[JointSpec, ...]
    materials: tuple[MaterialSpec, ...]

    def part(self, part_id: str) -> PartSpec:
        return _find(self.parts, part_id, "part")

    def joint(self, joint_id: str) -> JointSpec:
        return _find(self.joints, joint_id, "joint")

    def replace_part(self, part_id: str, **changes) -> AssetSpec:
        parts = tuple(dataclasses.replace(p, **changes) if p.id == part_id else p for p in self.parts)
        self.part(part_id)
        return dataclasses.replace(self, parts=parts)

    def replace_joint(self, joint_id: str, **changes) -> AssetSpec:
        joints = tuple(dataclasses.replace(j, **changes) if j.id == joint_id else j for j in self.joints)
        self.joint(joint_id)
        return dataclasses.replace(self, joints=joints)

    @property
    def part_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.parts)

    @property
    def joint_ids(self) -> tuple[str, ...]:
        return tuple(j.id for j in self.joints)


def _find(items, item_id: str, kind: str):
    for it in items:
        if it.id == item_id:
            return it
    raise KeyError(f"unknown {kind} id {item_id!r}")
