"""Structural target policy for complex Hard candidate objects."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

MINIMUM_ACTIVE_JOINTS = 3
MINIMUM_PARTS = 5
MINIMUM_REVOLUTE_TRAVEL_DEG = 60.0
MINIMUM_PRISMATIC_TRAVEL_DIAGONAL_FRACTION = 0.05
MINIMUM_TARGET_RELATIVE_AREA = 0.10
MINIMUM_TARGET_RELATIVE_LONG_EDGE = 0.20
EXCLUDED_CATEGORIES = set()
ALLOWED_ROLES = {'arm', 'board', 'door', 'drawer', 'lid', 'shelf', 'tray', 'window'}


def load_semantics(source_urdf: Path) -> dict[str, str]:
    path = Path(source_urdf).with_name('semantics.txt')
    rows = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) == 3:
            rows[fields[0]] = fields[2].strip().lower()
    return rows


def semantic_role(raw: str, category: str) -> str | None:
    role = raw.strip().lower()
    for prefix in ('rotation_', 'translation_'):
        if role.startswith(prefix):
            role = role[len(prefix):]
    if category == 'Table' and role == 'bar':
        role = 'board'
    if category == 'Lamp' and role == 'bar':
        role = 'arm'
    return role if role in ALLOWED_ROLES else None


def candidate_targets(spec, model, semantics: dict[str, str]) -> tuple[list[dict], list[str]]:
    reasons = []
    if spec.category in EXCLUDED_CATEGORIES:
        reasons.append('excluded_category')
    if len(spec.joints) < MINIMUM_ACTIVE_JOINTS:
        reasons.append('fewer_than_three_active_joints')
    if len(spec.parts) < MINIMUM_PARTS:
        reasons.append('fewer_than_five_parts')
    if reasons:
        return [], reasons

    geometry = model.world_vertices({})
    nonempty = [vertices for vertices in geometry.values() if len(vertices)]
    if not nonempty:
        return [], ['empty_rest_geometry']
    object_extents = np.ptp(np.concatenate(nonempty), axis=0)
    diagonal = float(np.linalg.norm(object_extents))
    long_edge = float(np.max(object_extents))
    area = float(np.prod(np.sort(object_extents)[-2:]))
    if min(diagonal, long_edge, area) <= 0 or not np.isfinite(object_extents).all():
        return [], ['invalid_rest_geometry']

    targets = []
    for joint in spec.joints:
        role = semantic_role(semantics.get(joint.part, ''), spec.category)
        if role is None:
            continue
        span = float(abs(joint.limits[1] - joint.limits[0]))
        kind = str(joint.type)
        if kind == 'revolute':
            travel = math.degrees(span)
            if travel < MINIMUM_REVOLUTE_TRAVEL_DEG:
                continue
            travel_fields = {'native_travel_deg': travel}
        elif kind == 'prismatic':
            travel_fraction = span / diagonal
            if travel_fraction < MINIMUM_PRISMATIC_TRAVEL_DIAGONAL_FRACTION:
                continue
            travel_fields = {'native_travel_m': span,
                             'native_travel_diagonal_fraction': travel_fraction}
        else:
            continue
        vertices = geometry.get(joint.part)
        if vertices is None or not len(vertices):
            continue
        extents = np.ptp(vertices, axis=0)
        relative_long_edge = float(np.max(extents) / long_edge)
        relative_area = float(np.prod(np.sort(extents)[-2:]) / area)
        if (relative_long_edge < MINIMUM_TARGET_RELATIVE_LONG_EDGE
                or relative_area < MINIMUM_TARGET_RELATIVE_AREA):
            continue
        targets.append({
            'joint_id': joint.id,
            'part_id': joint.part,
            'semantic_role': role,
            'joint_type': kind,
            'original_limits': list(joint.limits),
            'original_axis': list(joint.axis),
            'target_extents': extents.tolist(),
            'target_relative_long_edge': relative_long_edge,
            'target_relative_area': relative_area,
            **travel_fields,
        })
    if not targets:
        reasons.append('no_large_primary_motion_target')
    targets.sort(key=lambda row: (-row['target_relative_area'],
                                  -row['target_relative_long_edge'], row['joint_id']))
    return targets, reasons


def criteria() -> dict:
    return {
        'minimum_active_joints': MINIMUM_ACTIVE_JOINTS,
        'minimum_parts': MINIMUM_PARTS,
        'accepted_joint_types': ['revolute', 'prismatic'],
        'minimum_revolute_travel_degrees': MINIMUM_REVOLUTE_TRAVEL_DEG,
        'minimum_prismatic_travel_diagonal_fraction':
            MINIMUM_PRISMATIC_TRAVEL_DIAGONAL_FRACTION,
        'minimum_target_relative_area': MINIMUM_TARGET_RELATIVE_AREA,
        'minimum_target_relative_long_edge': MINIMUM_TARGET_RELATIVE_LONG_EDGE,
        'allowed_semantic_roles': sorted(ALLOWED_ROLES),
        'excluded_categories': sorted(EXCLUDED_CATEGORIES),
        'qualification': 'structural candidate only; render and human Hard review remain required',
    }
