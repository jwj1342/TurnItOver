"""Source-scoped native feature restoration and actual triangle contact checks.

Selectors come from construction provenance, never from a candidate verdict.
Unknown correspondence or exhausted geometry work is unknown, never success.
"""
from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from turnitover.checkers.base import PartMesh
from turnitover.checkers.native_surface import triangle_distance
from turnitover.checkers.mesh_heuristics import (
    HEURISTICS, contact_coverage, feature_match, same_surface, sampled_relation, scale,
    spatial_patch, surface_match,
)
from turnitover.repair.native_properties import join

FEATURE_DEFECTS = ('geometry.thin_sheet', 'geometry.missing_bevel')
CONTACT_DEFECTS = ('structure.floating', 'structure.interpenetration')
CONTACT_TOL = 1e-7  # Original native surface-contact acceptance, not fitted here.
SURFACE_TOL = 1e-8  # Original triangle-plane numerical tolerance.
MAX_PAIRS = 1_000_000


def _face_counter(faces):
    return Counter(tuple(sorted(map(int, face))) for face in faces)


def select_pair(reference, candidate, selector):
    """Map a frozen original component through stable native vertex records.

    Candidate face offsets may change after a repair. Select its faces through
    the reference component's vertex membership, rather than reusing offsets.
    """
    if not isinstance(selector, dict) or not isinstance(selector.get('part'), str):
        return None, None
    part = selector['part']
    if part not in reference:
        return None, None
    a = reference[part]
    if part not in candidate:
        return a, False
    b = candidate[part]
    ids = selector.get('face_ids')
    ranges = selector.get('vertex_ranges')
    if ids is not None:
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                or any(type(i) is not int or not 0 <= i < len(a.faces) for i in ids)):
            return None, None
        faces = a.faces[ids]
        vertices = np.unique(faces)
    elif ranges is not None:
        if (not isinstance(ranges, list) or not ranges
                or any(not isinstance(r, list) or len(r) != 2
                       or any(type(i) is not int for i in r)
                       or r[0] < 0 or r[1] <= 0 or sum(r) > len(a.vertices) for r in ranges)):
            return None, None
        vertices = np.unique(np.concatenate([np.arange(s, s + n) for s, n in ranges]))
        faces = a.faces[np.isin(a.faces, vertices).all(1)]
    else:
        return a, b
    if not len(faces):
        return None, None
    selected_reference = PartMesh(a.vertices, faces)
    if len(a.vertices) == len(b.vertices) and _face_counter(a.faces) == _face_counter(b.faces):
        selected = PartMesh(b.vertices, b.faces[np.isin(b.faces, vertices).all(1)])
    else:
        selected = spatial_patch(selected_reference, b)
    return selected_reference, selected


def _compact(mesh):
    ids, inverse = np.unique(mesh.faces, return_inverse=True)
    return PartMesh(np.asarray(mesh.vertices[ids], float), inverse.reshape(-1, 3))


def convex_certificate(mesh):
    """Sufficient convex-solid certificate on the actual exported triangles.

    Like the native solid readback, the checker-only support quotient welds
    exact positions, deduplicates coincident faces and orients one shell. Every
    oriented triangle must be a convex supporting plane; no hull replacement of
    a nonconvex/open input is accepted as a certificate.
    """
    import trimesh
    if not len(mesh.faces) or len(mesh.faces) > 5000:
        return None
    raw = _compact(mesh)
    vertices, inverse = np.unique(raw.vertices, axis=0, return_inverse=True)
    faces = inverse[raw.faces]
    _, keep = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    actual = trimesh.Trimesh(vertices, faces[np.sort(keep)], process=False)
    if not actual.is_watertight:
        return None
    actual.fix_normals()
    if not actual.is_winding_consistent or len(actual.split(only_watertight=False)) != 1:
        return None
    triangles = actual.triangles
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    if area2.min() <= 2e-12 or actual.volume <= 0:
        return None
    normals = cross / area2[:, None]
    if len(vertices) * len(normals) > MAX_PAIRS:
        return None
    # Positive orientation plus closed incidence and convex supports avoids
    # treating nested shells or a convex hull of a concave input as the object.
    if any(float(((vertices - tri[0]) @ normal).max()) > SURFACE_TOL
           for tri, normal in zip(triangles, normals, strict=True)):
        return None
    try:
        hull = ConvexHull(vertices)
    except QhullError:
        return None
    edges = np.unique(np.sort(np.concatenate([hull.simplices[:, [0, 1]],
        hull.simplices[:, [1, 2]], hull.simplices[:, [2, 0]]]), axis=1), axis=0)
    directions = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths == 0):
        return None
    return {'vertices': vertices, 'faces': actual.faces, 'normals': hull.equations[:, :3],
            'edges': directions / lengths[:, None], 'volume': float(actual.volume)}


def solid_relation(a, b):
    """Complete convex SAT: separated/contact/positive-volume overlap.

    Unlike counting surface crossings, this also detects full containment and
    coincident solids. AABB is not used as a positive verdict.
    """
    ca, cb = convex_certificate(a), convex_certificate(b)
    if ca is None or cb is None:
        return sampled_relation(a, b, max_pairs=MAX_PAIRS * 8)
    pair_count = len(ca['edges']) * len(cb['edges'])
    if pair_count > MAX_PAIRS:
        return {'relation': None, 'reason': 'Solid separating-axis work budget exceeded'}
    cross = np.cross(ca['edges'][:, None], cb['edges'][None]).reshape(-1, 3)
    lengths = np.linalg.norm(cross, axis=1)
    cross = cross[lengths > 0] / lengths[lengths > 0, None]
    axes = np.concatenate([ca['normals'], cb['normals'], cross])
    if len(axes) * (len(ca['vertices']) + len(cb['vertices'])) > MAX_PAIRS:
        return {'relation': None, 'reason': 'Solid projection work budget exceeded'}
    largest_gap = -math.inf
    for axis in axes:
        pa, pb = ca['vertices'] @ axis, cb['vertices'] @ axis
        gap = float(max(pa.min() - pb.max(), pb.min() - pa.max()))
        largest_gap = max(largest_gap, gap)
    relation = ('separated' if largest_gap > SURFACE_TOL else
                'overlap' if largest_gap < -SURFACE_TOL else 'contact')
    return {'relation': relation, 'maximum_separating_gap_m': largest_gap,
            'axes_checked': len(axes), 'tolerance_m': SURFACE_TOL,
            'certificate': 'actual_closed_convex_surfaces_all_face_normals_and_edge_cross_axes'}


def connection(a, b, tolerance=CONTACT_TOL):
    """Original triangle-feature distance, with explicit work-cap semantics."""
    if not len(a.faces) or not len(b.faces):
        return {'connected': False, 'reason': 'Measured empty contact surface'}
    ta, tb = a.vertices[a.faces], b.vertices[b.faces]
    if len(ta) * len(tb) > MAX_PAIRS:
        return {'connected': None, 'reason': 'Native surface-distance work budget exceeded'}
    lower = np.linalg.norm(np.maximum(0., np.maximum(ta.min(1)[:, None] - tb.max(1),
                                                    tb.min(1) - ta.max(1)[:, None])), axis=2)
    best = math.inf
    for flat in np.argsort(lower, axis=None):
        i, j = np.unravel_index(flat, lower.shape)
        if lower[i, j] > best:
            break
        best = min(best, triangle_distance(ta[i], tb[j], SURFACE_TOL))
        if best <= tolerance:
            break
    return {'connected': best <= tolerance, 'minimum_surface_distance_m': best,
            'tolerance_m': tolerance,
            'scope': 'native_geometric_assembly_connection_not_gravity_or_mechanical_stability'}


def _restoration(a, b, tolerance):
    if b is False:
        return False
    if a is None or b is None or not len(a.faces) or not len(b.faces):
        return None if a is None or b is None else False
    if len(a.vertices) != len(b.vertices):
        return None
    ids = np.unique(a.faces)
    error = float(np.linalg.norm(a.vertices[ids] - b.vertices[ids], axis=1).max())
    # A recovered native feature must restore its triangles, not just its box
    # or unused vertex records. Face offsets/order may change independently.
    needed, actual = _face_counter(a.faces), _face_counter(b.faces)
    return error <= tolerance and all(actual[f] >= n for f, n in needed.items())


def measure_geometry(kind, scope, histories, program_shas):
    result = {'checker_id': 'native.source_scoped_' + kind.split('.')[-1], 'passed': None}
    if (not isinstance(scope, dict) or scope.get('defect_id') != kind
            or not scope.get('provenance')
            or scope.get('reference_program_sha256') != program_shas.get('reference')):
        return {**result, 'reason': 'Missing hash-bound construction scope for this reference'}
    if (any(value is None for value in histories.values())
            or set(histories['reference']) != set(histories['candidate'])):
        return {**result, 'reason': 'Incomplete matching current native histories'}
    tolerance = scope.get('tolerance_m')
    if kind in FEATURE_DEFECTS and (type(tolerance) not in (int, float)
                                   or not math.isfinite(tolerance) or tolerance < 0):
        return {**result, 'reason': 'Missing original native feature tolerance'}
    outcomes, details = [], []
    for state in histories['reference']:
        ref, cand = histories['reference'][state], histories['candidate'][state]
        if ref is None or cand is None:
            outcomes.append(None)
            continue
        a, b = select_pair(ref, cand, scope.get('target'))
        if kind in FEATURE_DEFECTS:
            exact = _restoration(a, b, tolerance)
            detail = ({'passed': True, 'method': 'native_feature_restoration'} if exact is True else
                feature_match(a, b, kind, tolerance=tolerance, depth=scope.get('feature_depth_m')))
            passed = detail['passed']
            detail = {'state': state, **detail}
        else:
            x, y = select_pair(ref, cand, scope.get('support'))
            if a is None or x is None or b is None or y is None:
                passed, detail = None, {'state': state, 'reason': 'Native component correspondence unknown'}
            elif b is False or y is False:
                passed, detail = False, {'state': state, 'reason': 'Target/support removed'}
            elif kind == 'structure.floating':
                original = connection(a, x)
                tol = max(CONTACT_TOL, min(scale(a), scale(x)) * HEURISTICS['contact_relative_tolerance'])
                current = connection(b, y, tol)
                baseline = contact_coverage(a, x, tol)
                coverage = contact_coverage(b, y, tol)
                passed = join([current['connected'], None if baseline is None or coverage is None else
                    coverage >= baseline * HEURISTICS['contact_coverage_ratio']]) if original['connected'] is True else None
                detail = {'state': state, 'reference': original, 'candidate': current,
                          'reference_contact_coverage': baseline, 'candidate_contact_coverage': coverage}
            else:
                original, current = solid_relation(a, x), solid_relation(b, y)
                healthy = original['relation'] in {'separated', 'contact'}
                passed = (current['relation'] != 'overlap' if healthy and current['relation'] is not None else None)
                detail = {'state': state, 'reference': original, 'candidate': current}
        outcomes.append(passed)
        details.append(detail)
    return {**result, 'passed': join(outcomes), 'details': details,
            'construction_provenance': scope['provenance'],
            'limitations': ['Remeshed features and nonconvex contacts may use deterministic geometric heuristics.',
                'These checks do not certify general physical validity.']}


def contact_regressions(histories):
    """Check current pairs against reference assembly, allowing baseline intersections."""
    if (any(value is None for value in histories.values())
            or set(histories['reference']) != set(histories['candidate'])):
        return {'passed': None, 'checker_id': 'native.all_part_contacts'}
    outcomes = []
    for state, reference in histories['reference'].items():
        candidate = histories['candidate'][state]
        if reference is None or candidate is None or not reference or set(reference) != set(candidate):
            outcomes.append(None)
            continue
        if len(reference) < 2:
            outcomes.append(True)
        for left, right in itertools.combinations(reference, 2):
            if all(same_surface(reference[p], candidate[p]) for p in (left, right)):
                outcomes.append(True)
                continue
            original = solid_relation(reference[left], reference[right])
            current = solid_relation(candidate[left], candidate[right])
            if original['relation'] is None:
                outcomes.append(None)
            elif current['relation'] is None:
                outcomes.append(None)
            elif original['relation'] == 'overlap':
                if current['relation'] == 'separated':
                    outcomes.append(False)
                elif current['relation'] == 'contact':
                    outcomes.append(True)
                elif all('maximum_separating_gap_m' in row for row in (original, current)):
                    tolerance = max(CONTACT_TOL, min(scale(reference[left]), scale(reference[right])) *
                                    HEURISTICS['contact_relative_tolerance'])
                    outcomes.append(current['maximum_separating_gap_m'] >=
                                    original['maximum_separating_gap_m'] - tolerance)
                elif all('sampled_overlap_fraction' in row for row in (original, current)):
                    outcomes.append(current['sampled_overlap_fraction'] <=
                        original['sampled_overlap_fraction'] + HEURISTICS['overlap_fraction_tolerance'])
                else:
                    outcomes.append(None)
            else:
                outcomes.append(current['relation'] != 'overlap' and
                    (original['relation'] != 'contact' or current['relation'] == 'contact'))
    return {'passed': join(outcomes), 'checker_id': 'native.all_part_contacts',
            'scope': 'reference_relative_named_part_contacts_over_all_current_snapshots'}


def feature_regressions(histories):
    """Reference-relative surfaces allow subdivision and vertex renumbering."""
    if (any(value is None for value in histories.values())
            or set(histories['reference']) != set(histories['candidate'])):
        return {'passed': None, 'checker_id': 'native.full_native_feature_restoration'}
    values = []
    for state, reference in histories['reference'].items():
        candidate = histories['candidate'][state]
        if reference is None or candidate is None or not reference or set(reference) != set(candidate):
            values.append(None)
            continue
        for part, a in reference.items():
            b = candidate[part]
            matched = (a.vertices.shape == b.vertices.shape and np.array_equal(a.vertices, b.vertices)
                       and _face_counter(a.faces) == _face_counter(b.faces))
            values.append(True if matched else surface_match(a, b)['passed'])
    return {'passed': join(values), 'checker_id': 'native.reference_surface_features',
            'scope': 'bidirectional_surface_coverage_over_all_current_snapshots',
            'limitations': ['Area sampling and relative tolerances approximate surface equivalence.']}
