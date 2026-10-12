"""Strict UniPhys-40K ZIP to :class:`AssetSpec` conversion.

The released OBJ coordinates are normalized source coordinates, not declared
metres.  This importer therefore derives a per-asset uniform scale only when
an explicit centimetre annotation agrees with the corresponding part or full
model mesh extents on all three axes.  Ambiguous assets are rejected instead
of receiving a guessed scale or kinematic hierarchy.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import trimesh

from turnitover.assets.catalog import validate_spec
from turnitover.core.spec import AssetSpec, JointSpec, JointType, MaterialSpec, MeshSpec, PartSpec


PART_JSON = re.compile(r"annotations/part_(\d+)\.json$")
PART_OBJ = re.compile(r"parts/part_(\d+)\.obj$")
DIMENSIONS_CM = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*[*x×]\s*(\d+(?:\.\d+)?)\s*[*x×]\s*"
    r"(\d+(?:\.\d+)?)\s*cm\b"
)
SCALE_RELATIVE_TOLERANCE = 0.08


class UnsupportedAsset(ValueError):
    """The archive cannot be represented without an unverified assumption."""


@dataclass(frozen=True)
class ConvertedUniPhys40K:
    spec: AssetSpec
    report: dict


def _json_member(zf: zipfile.ZipFile, name: str) -> dict:
    try:
        value = json.loads(zf.read(name))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UnsupportedAsset(f"invalid JSON member: {name}") from exc
    if not isinstance(value, dict):
        raise UnsupportedAsset(f"JSON member must be an object: {name}")
    return value


def _single(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise UnsupportedAsset(f"expected one {suffix}, found {len(matches)}")
    return matches[0]


def _safe_extract(zf: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for info in zf.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise UnsupportedAsset(f"unsafe ZIP member: {info.filename}")
        target = (destination / Path(*member.parts)).resolve()
        if target != root and root not in target.parents:
            raise UnsupportedAsset(f"unsafe ZIP member: {info.filename}")
    zf.extractall(destination)


def _mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise UnsupportedAsset(f"empty or unsupported OBJ mesh: {path.name}")
    if mesh.faces.shape[1] != 3 or not np.isfinite(mesh.vertices).all():
        raise UnsupportedAsset(f"OBJ must contain finite triangular geometry: {path.name}")
    return mesh


def _canonical_triangles(meshes: list[trimesh.Trimesh]) -> np.ndarray:
    """Return triangle coordinates independent of vertex indexing and winding."""
    triangles = np.round(np.concatenate([mesh.triangles for mesh in meshes]), decimals=7)
    vertex_order = np.lexsort(
        (triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    triangles = np.take_along_axis(triangles, vertex_order[:, :, None], axis=1).reshape(-1, 9)
    triangle_order = np.lexsort(tuple(triangles[:, index] for index in reversed(range(9))))
    return triangles[triangle_order]


def _dimensions_cm(text: object) -> np.ndarray | None:
    if not isinstance(text, str) or not (match := DIMENSIONS_CM.search(text)):
        return None
    result = np.asarray([float(value) for value in match.groups()], dtype=float)
    return result if np.all(result > 0) else None


def _finite_vector(value: object, length: int, label: str) -> np.ndarray:
    if (not isinstance(value, list) or len(value) != length
            or any(type(item) not in (int, float) for item in value)):
        raise UnsupportedAsset(f"{label} must contain {length} numbers")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise UnsupportedAsset(f"{label} must be finite")
    return result


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected) / expected))


def _derive_scale(object_record: dict, annotations: dict[str, dict],
                  meshes: dict[str, trimesh.Trimesh], full_mesh: trimesh.Trimesh,
                  moving: set[str]) -> tuple[float, dict]:
    object_cm = np.sort(_finite_vector(object_record.get("volume"), 3, "object volume"))
    if np.any(object_cm <= 0):
        raise UnsupportedAsset("object dimensions must be positive")
    candidates = []
    for part_id, annotation in annotations.items():
        if part_id in moving:
            continue
        description_cm = _dimensions_cm((annotation.get("part_level") or {}).get("basic_description"))
        if description_cm is None:
            continue
        description_cm = np.sort(description_cm)
        extents = np.sort(np.asarray(meshes[part_id].extents, dtype=float))
        if np.any(extents <= 0):
            continue
        # The largest dimension is least sensitive to approximate descriptions
        # and thin-surface tessellation.  The other axes independently validate it.
        scale = float((description_cm[-1] / 100.0) / extents[-1])
        predicted_m = extents * scale
        expected_m = description_cm / 100.0
        error = _relative_error(predicted_m, expected_m)
        if error <= SCALE_RELATIVE_TOLERANCE:
            candidates.append((error, int(part_id), part_id, scale, extents, expected_m, predicted_m))
    if not candidates:
        extents = np.sort(np.asarray(full_mesh.extents, dtype=float))
        scale = float((object_cm[-1] / 100.0) / extents[-1])
        predicted_m = extents * scale
        expected_m = object_cm / 100.0
        error = _relative_error(predicted_m, expected_m)
        if error > SCALE_RELATIVE_TOLERANCE:
            raise UnsupportedAsset(
                "no part or full-model AABB validates a uniform OBJ-to-metre scale"
            )
        return scale, {
            "method": "object_annotation_dimensions_cm_vs_full_model_obj_aabb",
            "object_annotation_dimensions_cm": object_cm.tolist(),
            "obj_extents": extents.tolist(),
            "annotation_extents_m": expected_m.tolist(),
            "scaled_extents_m": predicted_m.tolist(),
            "max_relative_error": error,
            "metres_per_obj_unit": scale,
        }
    candidates.sort()
    error, _, part_id, scale, extents, expected_m, predicted_m = candidates[0]
    return scale, {
        "method": "fixed_part_annotation_dimensions_cm_vs_part_obj_aabb",
        "reference_part": part_id,
        "object_annotation_dimensions_cm": object_cm.tolist(),
        "reference_part_annotation_dimensions_cm": (expected_m * 100.0).tolist(),
        "obj_extents": extents.tolist(),
        "annotation_extents_m": expected_m.tolist(),
        "scaled_extents_m": predicted_m.tolist(),
        "max_relative_error": error,
        "metres_per_obj_unit": scale,
    }


def _motion(annotations: dict[str, dict]) -> tuple[str, JointType, np.ndarray, np.ndarray,
                                                        tuple[float, float], set[str]]:
    motions = []
    for part_id, annotation in annotations.items():
        kinematic = annotation.get("kinematic_info") or {}
        motion_info = kinematic.get("motion_info") or {}
        for code, joint_type in (("B", JointType.PRISMATIC), ("C", JointType.REVOLUTE)):
            if code in (kinematic.get("motion_types") or []) and isinstance(motion_info.get(code), dict):
                motions.append((part_id, joint_type, motion_info[code], motion_info.get("dependency")))
    if len(motions) != 1:
        raise UnsupportedAsset(f"exactly one independent B/C motion is supported, found {len(motions)}")
    owner, joint_type, raw, dependencies = motions[0]
    if (not isinstance(dependencies, list) or not dependencies
            or any(type(value) is not int or value <= 0 for value in dependencies)):
        raise UnsupportedAsset("motion dependency must be a nonempty integer part list")
    moving = {str(value) for value in dependencies}
    if len(moving) != len(dependencies) or owner not in moving or not moving <= set(annotations):
        raise UnsupportedAsset("motion dependency must uniquely include its owner and known parts")
    axis = _finite_vector(raw.get("axis"), 3, "joint axis")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise UnsupportedAsset("joint axis must be nonzero")
    pivot = _finite_vector(raw.get("pos"), 3, "joint pivot")
    travel = _finite_vector(raw.get("range"), 2, "joint range")
    if not travel[0] <= 0 <= travel[1] or not travel[0] < travel[1]:
        raise UnsupportedAsset("joint range must be nonzero and include rest=0")
    return owner, joint_type, axis / norm, pivot, (float(travel[0]), float(travel[1])), moving


def _pivot_evidence(pivot: np.ndarray, owner: str, moving: set[str],
                    meshes: dict[str, trimesh.Trimesh]) -> dict:
    overall = np.ptp(np.vstack([mesh.vertices for mesh in meshes.values()]), axis=0)
    tolerance = max(float(np.max(overall)) * 0.01, 1e-8)

    def contains(part_id: str) -> bool:
        bounds = meshes[part_id].bounds
        return bool(np.all(pivot >= bounds[0] - tolerance) and np.all(pivot <= bounds[1] + tolerance))

    static_contacts = sorted(part_id for part_id in meshes if part_id not in moving and contains(part_id))
    if not contains(owner) or not static_contacts:
        raise UnsupportedAsset("joint pivot is not supported by both owner and static-part bounds")
    return {"owner_contains_pivot": owner, "static_parts_containing_pivot": static_contacts,
            "aabb_tolerance_obj_units": tolerance}


def _mesh_spec(mesh: trimesh.Trimesh, *, scale: float, offset: np.ndarray) -> MeshSpec:
    vertices = (np.asarray(mesh.vertices, dtype=float) * scale) - offset
    colors: tuple[tuple[float, float, float], ...] = ()
    try:
        rgba = np.asarray(mesh.visual.to_color().vertex_colors)
        if rgba.shape == (len(vertices), 4):
            rgb = rgba[:, :3].astype(float) / 255.0
            rgb = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            colors = tuple(tuple(float(value) for value in row) for row in rgb)
    except (AttributeError, TypeError, ValueError):
        pass
    return MeshSpec(
        tuple(tuple(float(value) for value in row) for row in vertices),
        tuple(tuple(int(value) for value in row) for row in np.asarray(mesh.faces)),
        colors,
    )


def convert_archive(path: Path) -> ConvertedUniPhys40K:
    """Convert one verified single-joint UniPhys-40K archive."""
    path = path.resolve()
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        metadata_name = _single(names, "/meta_data.json")
        object_name = _single(names, "/annotations/object.json")
        prefix = metadata_name.removesuffix("meta_data.json")
        metadata = _json_member(zf, metadata_name)
        object_record = _json_member(zf, object_name)
        annotation_names = {match.group(1): name for name in names
                            if (match := PART_JSON.search(name))}
        mesh_names = {match.group(1): name for name in names if (match := PART_OBJ.search(name))}
        expected_parts = (metadata.get("geometry") or {}).get("num_parts")
        if (not isinstance(expected_parts, int) or expected_parts <= 0
                or set(annotation_names) != set(mesh_names)
                or len(annotation_names) != expected_parts):
            raise UnsupportedAsset("part OBJ/annotation set does not match metadata part count")
        annotations = {part_id: _json_member(zf, name) for part_id, name in annotation_names.items()}
        asset_id = metadata.get("id")
        if not isinstance(asset_id, str) or prefix != f"{asset_id}/":
            raise UnsupportedAsset("archive root and metadata asset ID disagree")
        source = metadata.get("source") or {}
        if not all(isinstance(source.get(key), str) and source[key].strip()
                   for key in ("dataset", "original_id", "license_ref")):
            raise UnsupportedAsset("source provenance is incomplete")

        owner, joint_type, axis, pivot, limits, moving = _motion(annotations)
        with tempfile.TemporaryDirectory(prefix="turnitover-uniphys40k-") as temp:
            _safe_extract(zf, Path(temp))
            meshes = {part_id: _mesh(Path(temp) / name)
                      for part_id, name in mesh_names.items()}
            full_mesh = _mesh(Path(temp) / _single(names, "/full_model/model.obj"))

        part_triangles = _canonical_triangles([meshes[key] for key in sorted(meshes, key=int)])
        full_triangles = _canonical_triangles([full_mesh])
        if (part_triangles.shape != full_triangles.shape
                or not np.array_equal(part_triangles, full_triangles)):
            raise UnsupportedAsset("part meshes do not close to the released full model")
        if not set(meshes) - moving:
            raise UnsupportedAsset("at least one static part is required")
        scale, scale_report = _derive_scale(object_record, annotations, meshes, full_mesh, moving)
        pivot_report = (_pivot_evidence(pivot, owner, moving, meshes)
                        if joint_type is JointType.REVOLUTE
                        else {"pivot_validation": "not_applicable_for_prismatic"})
        anchor = pivot * scale

        parts = []
        for part_id in sorted(meshes, key=int):
            is_owner = part_id == owner
            parent = None if part_id not in moving or is_owner else f"part_{owner}"
            offset = anchor if part_id in moving else np.zeros(3)
            parts.append(PartSpec(
                id=f"part_{part_id}", parent=parent,
                origin=tuple(float(value) for value in (anchor if is_owner else np.zeros(3))),
                mesh=_mesh_spec(meshes[part_id], scale=scale, offset=offset),
            ))
        scaled_limits = limits if joint_type is JointType.REVOLUTE else tuple(value * scale for value in limits)
        joint = JointSpec(
            id=f"joint_part_{owner}", part=f"part_{owner}", type=joint_type,
            axis=tuple(float(value) for value in axis),
            anchor=tuple(float(value) for value in anchor), limits=scaled_limits,
        )
        category = (metadata.get("object") or {}).get("category")
        if not isinstance(category, str) or not category:
            raise UnsupportedAsset("object category is missing")
        spec = AssetSpec(asset_id, category, tuple(parts), (joint,), (MaterialSpec("default"),))
        validate_spec(spec)
        full_bounds_m = full_mesh.bounds * scale
        report = {
            "asset_id": asset_id,
            "status": "structural_candidate",
            "archive": str(path),
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": source,
            "object_name": (metadata.get("object") or {}).get("object_name"),
            "category": category,
            "part_count": len(parts),
            "full_model_vertices": len(full_mesh.vertices),
            "full_model_bounds_m": full_bounds_m.tolist(),
            "scale_validation": scale_report,
            "joint_validation": {
                "owner_part": owner,
                "type": joint_type.value,
                "axis": axis.tolist(),
                "pivot_obj_units": pivot.tolist(),
                "pivot_m": anchor.tolist(),
                "source_range": list(limits),
                "asset_spec_limits": list(scaled_limits),
                "range_mapping": "source_qpos_relative_to_released_obj_zero_pose",
                "moving_parts": sorted(moving, key=int),
                "static_parts": sorted(set(meshes) - moving, key=int),
                **pivot_report,
            },
            "qualification": {
                "geometry_scale_and_grouping": "passed",
                "rendered_motion": "requires_review",
                "formal_train_validation_eligible": False,
            },
        }
        return ConvertedUniPhys40K(spec, report)
