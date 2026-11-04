"""Single-field, small-magnitude corruptions for full-motion Hard calibration."""
from __future__ import annotations

import dataclasses
import numpy as np
from scipy.spatial.transform import Rotation

from turnitover.core.program import ObjectProgram


def cone_directions(axis, angle_deg, variants=8):
    """Return deterministic unit axes at an exact angular distance."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
    perpendicular = basis - axis * (basis @ axis)
    perpendicular /= np.linalg.norm(perpendicular)
    result = []
    for azimuth in np.linspace(0, 360, variants, endpoint=False):
        direction = Rotation.from_rotvec(axis * np.deg2rad(azimuth)).apply(perpendicular)
        changed = axis * np.cos(np.deg2rad(angle_deg)) + direction * np.sin(np.deg2rad(angle_deg))
        result.append(tuple(float(v) for v in changed / np.linalg.norm(changed)))
    return result


def unit_directions(variants=8):
    """Deterministic well-separated directions for small translations."""
    base = np.asarray(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                       (0, 0, 1), (0, 0, -1), (1, 1, 1), (-1, 1, -1)), dtype=float)
    return [tuple(float(x) for x in row / np.linalg.norm(row)) for row in base[:variants]]


def joint_axis(program, joint_id, axis):
    spec = program.spec
    return ObjectProgram.from_spec(spec.replace_joint(joint_id, axis=tuple(axis)))


def joint_anchor(program, joint_id, delta):
    """Move only the pivot; the rest pose stays unchanged in the emitted ABI."""
    spec = program.spec
    joint = spec.joint(joint_id)
    anchor = tuple(float(a + d) for a, d in zip(joint.anchor, delta))
    return ObjectProgram.from_spec(spec.replace_joint(joint_id, anchor=anchor))


def part_translation(program, part_id, delta):
    spec = program.spec
    part = spec.part(part_id)
    origin = tuple(float(a + d) for a, d in zip(part.origin, delta))
    return ObjectProgram.from_spec(spec.replace_part(part_id, origin=origin))


def part_rotation(program, part_id, axis, angle_deg):
    """Rotate only the moving part's local rest orientation by an exact angle."""
    spec = program.spec
    part = spec.part(part_id)
    original = Rotation.from_euler('XYZ', part.rotation)
    changed = original * Rotation.from_rotvec(np.asarray(axis, dtype=float) * np.deg2rad(angle_deg))
    rotation = tuple(float(v) for v in changed.as_euler('XYZ'))
    return ObjectProgram.from_spec(spec.replace_part(part_id, rotation=rotation))


def changed_fields(reference, candidate):
    """Return dataclass field paths changed between two specs."""
    result = []
    for kind in ('parts', 'joints'):
        before = {row.id: row for row in getattr(reference.spec, kind)}
        after = {row.id: row for row in getattr(candidate.spec, kind)}
        for item_id in before:
            for field in dataclasses.fields(before[item_id]):
                if getattr(before[item_id], field.name) != getattr(after[item_id], field.name):
                    result.append(f'{kind}.{item_id}.{field.name}')
    return result
