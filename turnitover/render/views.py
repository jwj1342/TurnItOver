"""Discrete view set loaded from configs/views.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from turnitover.core.actions import ViewId


@dataclass(frozen=True)
class ViewDef:
    id: ViewId
    azimuth_deg: float
    elevation_deg: float
    projection: str
    distance_factor: float = 1.0

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "projection": self.projection,
            "distance_factor": self.distance_factor,
        }


def load_views(path: Path) -> tuple[ViewDef, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    views = tuple(ViewDef(id=ViewId(v["id"]), azimuth_deg=float(v["azimuth_deg"]), elevation_deg=float(v["elevation_deg"]),
                          projection=str(v["projection"]), distance_factor=float(v.get("distance_factor", 1.0)))
                  for v in raw["views"])
    ids = [v.id for v in views]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate view ids")
    return views
