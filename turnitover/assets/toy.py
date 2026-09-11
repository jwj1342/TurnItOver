"""Hand-written toy asset: a cabinet with two prismatic drawers and one revolute door.

Frame: Y up, +Z toward the viewer (front). Body outer size 0.6 (X) x 0.8 (Y) x 0.5 (Z).
"""
from __future__ import annotations

import math
from typing import Iterator

from turnitover.core.spec import AssetSpec, JointSpec, JointType, MaterialSpec, PartSpec

W, H, D = 0.6, 0.8, 0.5
T = 0.02  # panel thickness

_MATERIALS = (
    MaterialSpec(id="body", color="#b08a5a", roughness=0.7),
    MaterialSpec(id="drawer", color="#5a7fb0", roughness=0.6),
    MaterialSpec(id="door", color="#8a5a5a", roughness=0.6),
)


def _panel(pid: str, origin, size) -> PartSpec:
    return PartSpec(id=pid, parent=None, origin=origin, size=size, material="body")


def toy_cabinet() -> AssetSpec:
    shelf_y = H * 0.5  # door bay above, drawer bays below
    bay_h = shelf_y / 2
    parts = (
        _panel("body_back", (0.0, H / 2, -D / 2 + T / 2), (W, H, T)),
        _panel("body_left", (-W / 2 + T / 2, H / 2, 0.0), (T, H, D)),
        _panel("body_right", (W / 2 - T / 2, H / 2, 0.0), (T, H, D)),
        _panel("body_top", (0.0, H - T / 2, 0.0), (W, T, D)),
        _panel("body_bottom", (0.0, T / 2, 0.0), (W, T, D)),
        _panel("body_shelf", (0.0, shelf_y, 0.0), (W - 2 * T, T, D - T)),
        PartSpec(
            id="drawer_0",
            parent=None,
            origin=(0.0, T + bay_h / 2, 0.0),
            size=(W - 2 * T - 0.01, bay_h - 0.02, D - T - 0.01),
            material="drawer",
        ),
        PartSpec(
            id="drawer_1",
            parent=None,
            origin=(0.0, T + bay_h + bay_h / 2, 0.0),
            size=(W - 2 * T - 0.01, bay_h - 0.02, D - T - 0.01),
            material="drawer",
        ),
        PartSpec(
            id="door",
            parent=None,
            origin=(0.0, shelf_y + (H - shelf_y) / 2, D / 2 - T / 2),
            size=(W - 2 * T - 0.01, H - shelf_y - T - 0.01, T),
            material="door",
        ),
    )
    joints = (
        JointSpec(id="drawer_0", part="drawer_0", type=JointType.PRISMATIC, axis=(0.0, 0.0, 1.0),
                  anchor=(0.0, T + bay_h / 2, 0.0), limits=(0.0, 0.35)),
        JointSpec(id="drawer_1", part="drawer_1", type=JointType.PRISMATIC, axis=(0.0, 0.0, 1.0),
                  anchor=(0.0, T + bay_h + bay_h / 2, 0.0), limits=(0.0, 0.35)),
        JointSpec(id="door", part="door", type=JointType.REVOLUTE, axis=(0.0, 1.0, 0.0),
                  anchor=(-W / 2 + T, shelf_y + (H - shelf_y) / 2, D / 2 - T / 2), limits=(0.0, math.pi / 2),
                  sign=-1.0),
    )
    return AssetSpec(asset_id="toy_cabinet", category="cabinet", parts=parts, joints=joints, materials=_MATERIALS)


class ToyAssetSource:
    source_id = "toy"

    def __iter__(self) -> Iterator[AssetSpec]:
        yield toy_cabinet()

    def get(self, asset_id: str) -> AssetSpec:
        spec = toy_cabinet()
        if asset_id != spec.asset_id:
            raise KeyError(asset_id)
        return spec
