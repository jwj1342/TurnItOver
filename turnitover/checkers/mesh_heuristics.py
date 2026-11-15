"""Deterministic, reference-relative repair heuristics on exported triangles.

Engineering defaults, not Validation-fitted thresholds or a physical proof.
No pose/scale registration, source program execution, or optional dependencies.
"""
from __future__ import annotations

import numpy as np
import trimesh

from turnitover.checkers.base import PartMesh
from turnitover.checkers.meshops import point_surface_distance, sample_surface

HEURISTICS = {
    'version': 1, 'status': 'development', 'seed': 0, 'surface_samples': 512,
    'surface_relative_tolerance': .005, 'surface_coverage': .95,
    'thickness_ratio': [.75, 1.25], 'bevel_depth_fraction': .25,
    'normal_cosine': .95, 'normal_coverage': .9,
    'contact_relative_tolerance': .002, 'contact_coverage_ratio': .5,
    'overlap_fraction_tolerance': .05,
    'occupancy_grid': 7, 'ray_agreement': .9, 'max_faces': 50000,
    'max_ray_pairs': 8_000_000,
}


def triangles(mesh):
    tri = np.asarray(mesh.vertices[mesh.faces], dtype=float)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    return tri, lengths / 2, np.divide(cross, lengths[:, None],
        out=np.zeros_like(cross), where=lengths[:, None] > 0)


def scale(mesh):
    return float(np.linalg.norm(np.ptp(mesh.vertices[np.unique(mesh.faces)], axis=0)))


def usable(mesh):
    return (isinstance(mesh, PartMesh) and 0 < len(mesh.faces) <= HEURISTICS['max_faces']
            and triangles(mesh)[1].sum() > 0)


def samples(mesh):
    return sample_surface(mesh.vertices, mesh.faces, HEURISTICS['surface_samples'],
                          np.random.default_rng(HEURISTICS['seed']), return_face_ids=True)


def same_surface(a, b):
    """Fast exact triangle-position multiset match, independent of numbering."""
    if a.faces.shape != b.faces.shape:
        return False
    if np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces):
        return True
    def canonical(mesh):
        tri = np.sort(mesh.vertices[mesh.faces].view(
            np.dtype((np.void, mesh.vertices.dtype.itemsize * 3))).reshape(-1, 3), axis=1)
        return np.sort(tri.copy().view(np.dtype((np.void, tri.dtype.itemsize * 3))).ravel())
    # Float32 browser exports and float64 fixtures must compare geometrically.
    left = PartMesh(np.asarray(a.vertices, dtype='<f8'), a.faces)
    right = PartMesh(np.asarray(b.vertices, dtype='<f8'), b.faces)
    return np.array_equal(canonical(left), canonical(right))


def surface_match(a, b, tolerance=None, *, normals=False):
    if not usable(a) or not usable(b):
        return {'passed': None, 'reason': 'Missing, degenerate or oversized surface'}
    tolerance = max(1e-7, scale(a) * HEURISTICS['surface_relative_tolerance']) if tolerance is None else tolerance
    if same_surface(a, b):
        return {'passed': True, 'surface_error_m': 0., 'tolerance_m': tolerance, 'method': 'same_surface'}
    errors, agreements = [], []
    for reference, candidate in ((a, b), (b, a)):
        points, ids = samples(reference)
        distance, nearest = point_surface_distance(points, candidate.vertices, candidate.faces, return_face_ids=True)
        errors.append(float(np.quantile(distance, HEURISTICS['surface_coverage'])))
        if normals:
            left, right = triangles(reference)[2][ids], triangles(candidate)[2][nearest]
            agreements.append(float(np.mean(abs(np.einsum('ij,ij->i', left, right)) >= HEURISTICS['normal_cosine'])))
    passed = max(errors) <= tolerance and (not normals or min(agreements) >= HEURISTICS['normal_coverage'])
    return {'passed': bool(passed), 'surface_error_m': max(errors), 'tolerance_m': tolerance,
            'normal_coverage': min(agreements) if agreements else None, 'method': 'sampled_bidirectional_surface'}


def feature_match(a, b, kind, *, tolerance=0., depth=None):
    if b is False or isinstance(b, PartMesh) and not len(b.faces):
        return {'passed': False, 'reason': 'Selected feature removed'}
    if not usable(a) or not usable(b):
        return {'passed': None, 'reason': 'Feature surface unavailable'}
    if kind == 'geometry.thin_sheet':
        matched = surface_match(a, b)
        points, _ = samples(a)
        axes = np.linalg.eigh(np.cov(points.T))[1]
        left = np.diff(np.quantile(points @ axes, [.01, .99], axis=0), axis=0)[0]
        right = np.diff(np.quantile(samples(b)[0] @ axes, [.01, .99], axis=0), axis=0)[0]
        if left.min() <= max(1e-10, scale(a) * 1e-8):
            return {'passed': None, 'reason': 'Reference has no measurable thickness'}
        ratios = right / left
        lower, upper = HEURISTICS['thickness_ratio']
        return {**matched, 'passed': bool(matched['passed'] and np.all((ratios >= lower) & (ratios <= upper))),
                'thickness_ratios': ratios.tolist(), 'thickness_ratio_bounds': [lower, upper]}
    threshold = max(tolerance, (depth * HEURISTICS['bevel_depth_fraction'])
                    if depth is not None else scale(a) * .001, 1e-7)
    matched = surface_match(a, b, threshold, normals=True)
    # Oversample minor reference normal groups so a narrow bevel band cannot
    # disappear into the dominant flat faces' global average.
    _, areas, normals = triangles(a)
    _, groups = np.unique(np.round(normals, 4), axis=0, return_inverse=True)
    weights = np.bincount(groups, weights=areas)
    feature = weights[groups] < weights.max() * .25
    if feature.any():
        patch = PartMesh(a.vertices, a.faces[feature])
        points, ids = samples(patch)
        distance, nearest = point_surface_distance(points, b.vertices, b.faces, return_face_ids=True)
        alignment = abs(np.einsum('ij,ij->i', triangles(patch)[2][ids], triangles(b)[2][nearest]))
        coverage = float(np.mean((distance <= threshold) & (alignment >= HEURISTICS['normal_cosine'])))
        matched.update(feature_coverage=coverage,
                       passed=bool(matched['passed'] and coverage >= HEURISTICS['normal_coverage']))
    return matched


def spatial_patch(reference, candidate):
    """Reference-frozen spatial region fallback after native topology changes."""
    points = reference.vertices[np.unique(reference.faces)]
    margin = max(1e-7, scale(reference) * .05)
    tri = candidate.vertices[candidate.faces]
    selected = np.all(tri.max(1) >= points.min(0) - margin, axis=1) & np.all(
        tri.min(1) <= points.max(0) + margin, axis=1)
    return PartMesh(candidate.vertices, candidate.faces[selected])


def contact_coverage(a, b, tolerance):
    if not usable(a) or not usable(b):
        return None
    return max(float(np.mean(point_surface_distance(samples(left)[0], right.vertices,
        right.faces) <= tolerance)) for left, right in ((a, b), (b, a)))


def _inside(mesh, points):
    """Three deterministic ray parities; deduplicate shared-edge intersections."""
    tri = triangles(mesh)[0]
    edge1, edge2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    votes = []
    for direction in np.array([[1., .371, .173], [.193, 1., .419], [.317, .137, 1.]]):
        direction /= np.linalg.norm(direction)
        h = np.cross(direction, edge2)
        det = np.einsum('ij,ij->i', edge1, h)
        inv = np.divide(1., det, out=np.zeros_like(det), where=abs(det) > 1e-14)
        counts = []
        for point in points:
            delta = point - tri[:, 0]
            u = np.einsum('ij,ij->i', delta, h) * inv
            q = np.cross(delta, edge1)
            v = (q @ direction) * inv
            distance = np.einsum('ij,ij->i', edge2, q) * inv
            hits = np.sort(distance[(abs(det) > 1e-14) & (u >= -1e-10) & (v >= -1e-10)
                                    & (u + v <= 1 + 1e-10) & (distance > 1e-8)])
            counts.append((0 if not len(hits) else 1 + np.sum(np.diff(hits) > 1e-8)) % 2)
        votes.append(counts)
    votes = np.asarray(votes, dtype=bool)
    return votes.sum(0) >= 2, np.all(votes == votes[:1], axis=0)


def sampled_relation(a, b, *, max_pairs=8_000_000):
    result = {'relation': None, 'method': 'heuristic_multiray_occupancy', 'physical_certificate': False}
    if not usable(a) or not usable(b):
        return {**result, 'reason': 'Missing, degenerate or oversized contact surface'}
    va, vb = a.vertices[np.unique(a.faces)], b.vertices[np.unique(b.faces)]
    lower, upper = np.maximum(va.min(0), vb.min(0)), np.minimum(va.max(0), vb.max(0))
    if np.any(lower > upper + 1e-8):
        return {**result, 'relation': 'separated', 'method': 'disjoint_bounds'}
    n = HEURISTICS['occupancy_grid']
    axes = [lo + (np.arange(n) + .5) / n * (hi - lo) for lo, hi in zip(lower, upper)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    pa, pb = samples(a)[0], samples(b)[0]
    if 3 * (len(grid) + len(pa) + len(pb)) * (len(a.faces) + len(b.faces)) > min(max_pairs, HEURISTICS['max_ray_pairs']):
        return {**result, 'reason': 'Occupancy sampling work budget exceeded'}
    points = np.concatenate([grid, pa, pb])
    ia, qa = _inside(a, points)
    ib, qb = _inside(b, points)
    da = point_surface_distance(points, a.vertices, a.faces)
    db = point_surface_distance(points, b.vertices, b.faces)
    agreement = float(np.mean(qa & qb))
    if agreement < HEURISTICS['ray_agreement']:
        return {**result, 'reason': 'Ambiguous open-surface ray votes', 'ray_agreement': agreement}
    interior = (ia & ib & qa & qb & (da > 1e-7) & (db > 1e-7))
    # A point on one surface but strictly inside the other also proves sampled
    # penetration, including thin overlaps missed by the volume grid.
    surface_overlap = np.any(ib[len(grid):len(grid) + len(pa)] & qb[len(grid):len(grid) + len(pa)]
                             & (db[len(grid):len(grid) + len(pa)] > 1e-7)) or np.any(
        ia[-len(pb):] & qa[-len(pb):] & (da[-len(pb):] > 1e-7))
    near = min(float(da[-len(pb):].min()), float(db[len(grid):len(grid) + len(pa)].min()))
    relation = 'overlap' if interior.any() or surface_overlap else 'contact' if near <= 1e-7 else 'separated'
    return {**result, 'relation': relation, 'ray_agreement': agreement,
            'sampled_overlap_fraction': float(np.mean(interior[:len(grid)])), 'samples': len(points)}


def component_count(mesh):
    """Count substantial exact-position-connected surfaces, including shifted clones."""
    vertices, ids = np.unique(mesh.vertices, axis=0, return_inverse=True)
    actual = trimesh.Trimesh(vertices, ids[mesh.faces], process=False)
    components = actual.split(only_watertight=False)
    return int(sum(c.area >= actual.area * .001 for c in components))
