"""Defect taxonomy loaded from ``defects.yaml``; ids are the label vocabulary."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from importlib import resources

import yaml


class Layer(StrEnum):
    GEOMETRY = "geometry"
    STRUCTURE = "structure"
    KINEMATICS = "kinematics"
    MATERIAL = "material"
    RUNTIME = "runtime"


class Visibility(StrEnum):
    STATIC_VISIBLE = "static_visible"
    VIEW_DEPENDENT = "view_dependent"
    ANIMATION_REQUIRED = "animation_required"
    PROBE_REQUIRED = "probe_required"
    TEMPORAL_REQUIRED = "temporal_required"
    CONTEXT_DEPENDENT = "context_dependent"


@dataclass(frozen=True)
class DefectDef:
    id: str
    layer: Layer
    name_zh: str
    injection: str
    visibility: Visibility
    visibility_note: str
    label_form: str
    implemented: bool


@dataclass(frozen=True)
class Taxonomy:
    schema_version: int
    label_forms: tuple[str, ...]
    defects: tuple[DefectDef, ...]

    def by_id(self, defect_id: str) -> DefectDef:
        for d in self.defects:
            if d.id == defect_id:
                return d
        raise KeyError(f"unknown defect id {defect_id!r}")

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(d.id for d in self.defects)

    def implemented_ids(self) -> tuple[str, ...]:
        return tuple(d.id for d in self.defects if d.implemented)


def _raw() -> dict:
    text = resources.files(__package__).joinpath("defects.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


@cache
def load_taxonomy() -> Taxonomy:
    raw = _raw()
    label_forms = tuple(raw["label_forms"])
    defects = []
    for row in raw["defects"]:
        layer = Layer(row["layer"])
        if not row["id"].startswith(f"{layer.value}."):
            raise ValueError(f"defect id {row['id']} must be prefixed by its layer {layer.value}")
        if row["label_form"] not in label_forms:
            raise ValueError(f"defect {row['id']} has unknown label_form {row['label_form']}")
        defects.append(
            DefectDef(
                id=row["id"],
                layer=layer,
                name_zh=row["name_zh"],
                injection=row["injection"],
                visibility=Visibility(row["visibility"]),
                visibility_note=row["visibility_note"],
                label_form=row["label_form"],
                implemented=bool(row["implemented"]),
            )
        )
    ids = [d.id for d in defects]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate defect ids in taxonomy")
    return Taxonomy(schema_version=int(raw["schema_version"]), label_forms=label_forms, defects=tuple(defects))
