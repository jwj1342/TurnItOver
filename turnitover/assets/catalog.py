"""Local, provenance-bearing AssetSpec catalog. Mesh/URDF conversion is a separate step."""
from __future__ import annotations

import json
import math
import re
from dataclasses import fields
from pathlib import Path

from turnitover.core.serde import from_dict
from turnitover.core.spec import AssetSpec, JointSpec, MaterialSpec, MeshSpec, PartSpec


def decode_spec(raw: dict) -> AssetSpec:
    # The generic serde accepts extra fields for backwards-compatible records.
    # Catalog ingestion must reject unsupported mesh fields instead of making boxes silently.
    for tp, rows in ((AssetSpec, [raw]), (PartSpec, raw.get("parts", [])),
                     (JointSpec, raw.get("joints", [])), (MaterialSpec, raw.get("materials", []))):
        allowed = {f.name for f in fields(tp)} | {"__type__"}
        for row in rows:
            if not isinstance(row, dict) or set(row) - allowed:
                raise ValueError(f"Unsupported fields in {tp.__name__}")
            for key in ("origin", "rotation", "frame_rotation", "size", "segments", "axis", "anchor", "limits"):
                if key in row and (not isinstance(row[key], list) or len(row[key]) != (2 if key == "limits" else 3)):
                    raise ValueError(f"Invalid vector length: {key}")
            if row.get("mesh") is not None:
                mesh = row["mesh"]
                if not isinstance(mesh, dict) or set(mesh) - ({f.name for f in fields(MeshSpec)} | {"__type__"}):
                    raise ValueError("Unsupported fields in MeshSpec")
                for key in ("vertices", "faces", "colors"):
                    if any(not isinstance(v, list) or len(v) != 3 for v in mesh.get(key, [])):
                        raise ValueError("Invalid mesh vector length")
    spec = from_dict(AssetSpec, raw)
    validate_spec(spec)
    return spec


def validate_spec(spec: AssetSpec) -> None:
    def ids(values):
        names = [v.id for v in values]
        if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", n) for n in names):
            raise ValueError("IDs must be unique, nonempty and use letters, digits, dot, dash or underscore")

    def vector(v, length=3):
        if len(v) != length or any(type(x) not in (int, float) or not math.isfinite(x) for x in v):
            raise ValueError("Invalid finite vector")

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec.asset_id) or not spec.category or not spec.parts:
        raise ValueError("Asset needs a safe ID, category and parts")
    ids(spec.parts)
    ids(spec.joints)
    ids(spec.materials)
    materials = {m.id for m in spec.materials}
    parents = {p.id: p.parent for p in spec.parts}
    for p in spec.parts:
        for v in (p.origin, p.rotation, p.size, p.segments):
            vector(v)
        if any(x <= 0 for x in p.size) or any(type(x) is not int or x < 1 for x in p.segments):
            raise ValueError("Box sizes and integer subdivisions must be positive")
        if p.material not in materials:
            raise ValueError(f"Unknown material: {p.material}")
        if p.mesh is not None:
            for v in p.mesh.vertices:
                vector(v)
            for f in p.mesh.faces:
                if len(f) != 3 or any(type(i) is not int or not 0 <= i < len(p.mesh.vertices) for i in f):
                    raise ValueError("Mesh face has invalid vertex indices")
            if bool(p.mesh.faces) != bool(p.mesh.vertices):
                raise ValueError("Mesh must have both vertices and faces, or neither for an empty link")
            if p.mesh.colors and len(p.mesh.colors) != len(p.mesh.vertices):
                raise ValueError("Mesh colors must match vertices")
            for c in p.mesh.colors:
                vector(c)
                if any(not 0 <= x <= 1 for x in c):
                    raise ValueError("Mesh color out of range")
        seen, node = set(), p.id
        while node is not None:
            if node not in parents or node in seen:
                raise ValueError("Unknown parent or cycle in part hierarchy")
            seen.add(node)
            node = parents[node]
    if len({j.part for j in spec.joints}) != len(spec.joints):
        raise ValueError("Only one joint per part is supported")
    for j in spec.joints:
        vector(j.axis)
        vector(j.anchor)
        vector(j.frame_rotation)
        vector(j.limits, 2)
        if j.part not in parents or sum(x*x for x in j.axis) <= 0 or not j.limits[0] <= 0 <= j.limits[1]:
            raise ValueError("Joint needs a known part, nonzero axis and limits including rest=0")
        if j.sign not in (-1, 1):
            raise ValueError("Joint sign must be -1 or 1")
    for m in spec.materials:
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", m.color) or not 0 <= m.metalness <= 1 or not 0 <= m.roughness <= 1:
            raise ValueError("Materials need hex RGB and metalness/roughness in [0, 1]")


class CatalogSource:
    def __init__(self, manifest: str):
        path = Path(manifest)
        raw = json.loads(path.read_text())
        if raw.get("version") != 1 or not raw.get("source_id") or not raw.get("assets"):
            raise ValueError("Catalog requires version=1, source_id and nonempty assets")
        self.source_id = raw["source_id"]
        self._specs, self._metadata = {}, {}
        for entry in raw["assets"]:
            if any(not isinstance(entry.get(k), str) or not entry[k].strip()
                   for k in ("spec", "split_group", "origin", "license")):
                raise ValueError("Each asset requires spec, split_group, origin and license strings")
            spec = decode_spec(json.loads((path.parent / entry["spec"]).read_text()))
            if spec.asset_id in self._specs:
                raise ValueError(f"Duplicate asset ID: {spec.asset_id}")
            self._specs[spec.asset_id] = spec
            self._metadata[spec.asset_id] = {k: entry[k] for k in ("split_group", "origin", "license")}

    def __iter__(self):
        return iter(self._specs[k] for k in sorted(self._specs))

    def get(self, asset_id: str) -> AssetSpec:
        return self._specs[asset_id]

    def metadata(self, asset_id: str) -> dict:
        return dict(self._metadata[asset_id])
