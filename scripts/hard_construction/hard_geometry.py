"""World vertices and camera projection for immutable Hard candidate geometry."""
from __future__ import annotations

from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from turnitover.checkers.base import PartMesh
from turnitover.core.sample import Framing


def spec_geometry(spec, joint_values=None):
    """World vertices reconstructed from the emitted ABI (boxes use eight corners).

    Intended for bounds/displacement audit, not the sampled surface checker.
    """
    joint_values = joint_values or {}
    parts = {p['id']: p for p in spec['parts']}
    joints = {j['part']: j for j in spec['joints']}
    transforms = {}
    def transform(pid):
        if pid in transforms:
            return transforms[pid]
        p, local = parts[pid], np.eye(4)
        local[:3, :3] = Rotation.from_euler('XYZ', p['rotation']).as_matrix()
        local[:3, 3] = p['origin']
        if pid in joints:
            j, pivot = joints[pid], np.eye(4)
            pivot[:3, :3] = Rotation.from_euler('XYZ', j.get('frame_rotation', [0., 0., 0.])).as_matrix()
            pivot[:3, 3] = j['anchor']
            value = np.clip(joint_values.get(j['id'], 0.), *j['limits']) * j.get('sign', 1.)
            axis = np.asarray(j['axis']) / np.linalg.norm(j['axis'])
            if j['type'] == 'revolute':
                pivot[:3, :3] = pivot[:3, :3] @ Rotation.from_rotvec(axis * value).as_matrix()
            else:
                pivot[:3, 3] += pivot[:3, :3] @ (axis * value)
            local[:3, 3] -= j['anchor']
            local = pivot @ local
        transforms[pid] = (transform(p['parent']) @ local if p['parent'] else local)
        return transforms[pid]
    geometry = {}
    for pid, p in parts.items():
        if p.get('mesh') is not None:
            vertices = np.asarray(p['mesh']['vertices'], dtype=float)
            if not len(vertices):
                continue
        else:
            vertices = np.asarray(list(product((-.5, .5), repeat=3))) * p['size']
        matrix = transform(pid)
        geometry[pid] = PartMesh(vertices @ matrix[:3, :3].T + matrix[:3, 3], np.empty((0, 3), dtype=int))
    return geometry


def spec_diagonal(spec):
    """Rest world bounding-box diagonal, independent of supplied corruption labels."""
    points = np.concatenate([m.vertices for m in spec_geometry(spec).values()])
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError('Invalid reference bounding box')
    return diagonal


def normal_framing(rendered, padding):
    """Freeze framing from the complete normal trajectory, before any corruption."""
    points = np.concatenate([m.vertices for geometry in rendered['geometry'].values() for m in geometry.values()])
    low, high = points.min(axis=0), points.max(axis=0)
    radius = float(np.linalg.norm(high - low) / 2 * padding)
    if not np.isfinite(radius) or radius <= 0 or padding < 1:
        raise ValueError('Invalid normal-only framing')
    return Framing(tuple(float(v) for v in (low + high) / 2), radius)


def project_vertices(vertices, framing, view):
    """Use the same fixed camera formula as the observation harness, in pixels."""
    az, el = np.deg2rad([view.azimuth_deg, view.elevation_deg])
    direction = np.array([np.sin(az)*np.cos(el), np.sin(el), np.cos(az)*np.cos(el)])
    up = np.array([0., 0., -1.]) if abs(view.elevation_deg) > 89 else np.array([0., 1., 0.])
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    points = np.asarray(vertices) - framing.center
    if view.projection == 'orthographic':
        xy = np.column_stack((points @ right, points @ up)) / (1.1 * framing.radius)
    else:
        distance = framing.radius * (view.distance_factor or 1.) / np.sin(np.deg2rad(20)) * 1.05
        depth = distance - points @ direction
        xy = np.column_stack((points @ right, points @ up)) / (depth[:, None] * np.tan(np.deg2rad(20)))
    return xy * 128
