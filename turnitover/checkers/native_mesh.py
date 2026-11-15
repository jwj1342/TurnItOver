"""Native RP29 mesh measurements, independent of specs and corruption code.

The edge bookkeeping and plane-cut predicate are adapted from the original
native construction readbacks (hashes in docs/native-scoring-provenance.json).
This is deliberately NOT a general manifold/physical-solid certificate.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def topology(mesh):
    v, f = mesh.vertices, mesh.faces
    _, ids = np.unique(v, axis=0, return_inverse=True)
    faces = ids[f]
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, incidence = np.unique(edges, axis=0, return_counts=True)
    _, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    tri = v[f]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return {'boundary_edges': int((incidence == 1).sum()),
            'nonmanifold_edges': int((incidence > 2).sum()),
            'degenerate_faces': int((area2 < 1e-14).sum()),
            'duplicate_unoriented_faces': int((counts - 1).sum()),
            'triangles': len(f), 'vertices': len(v)}


def plane_cut(tri, distances, tol):
    points = [tri[i] for i in range(3) if abs(distances[i]) <= tol]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        if distances[i] * distances[j] < 0:
            points.append(tri[i] + (tri[j] - tri[i]) * distances[i] / (distances[i] - distances[j]))
    return points


def interior_barycentric(point, tri, tol):
    e0, e1 = tri[1] - tri[0], tri[2] - tri[0]
    delta = point - tri[0]
    gram = np.array([[e0 @ e0, e0 @ e1], [e0 @ e1, e1 @ e1]])
    try:
        u, v = np.linalg.solve(gram, np.array([delta @ e0, delta @ e1]))
    except np.linalg.LinAlgError:
        return False
    margin = tol / max(np.linalg.norm(e0), np.linalg.norm(e1), tol)
    return min(u, v, 1 - u - v) > margin


def proper_crossing(a, b, tol):
    na = np.cross(a[1] - a[0], a[2] - a[0])
    nb = np.cross(b[1] - b[0], b[2] - b[0])
    if min(np.linalg.norm(na), np.linalg.norm(nb)) < tol * tol:
        return False
    na /= np.linalg.norm(na)
    nb /= np.linalg.norm(nb)
    da, db = (a - b[0]) @ nb, (b - a[0]) @ na
    if da.min() > tol or da.max() < -tol or db.min() > tol or db.max() < -tol:
        return False
    line = np.cross(na, nb)
    length = np.linalg.norm(line)
    if length < 1e-8:
        return False
    line /= length
    pa, pb = plane_cut(a, da, tol), plane_cut(b, db, tol)
    if len(pa) < 2 or len(pb) < 2:
        return False
    ia, ib = np.asarray(pa) @ line, np.asarray(pb) @ line
    lower, upper = max(ia.min(), ib.min()), min(ia.max(), ib.max())
    if upper - lower <= tol * 4:
        return False
    origin = np.linalg.solve(np.stack((na, nb, line)), np.array([na @ a[0], nb @ b[0], 0.]))
    point = origin + line * ((lower + upper) / 2)
    return bool(interior_barycentric(point, a, tol) and interior_barycentric(point, b, tol))


def self_crossings(mesh, *, max_faces=10000, max_pairs=1000000):
    """Original proper nonadjacent/noncoplanar scope, with explicit work cap.

    Exhaustion is unknown, never zero. Neither deleting duplicate faces nor
    welding source geometry is allowed; tolerance remapping only skips adjacency.
    """
    v, f = mesh.vertices, mesh.faces
    if len(f) > max_faces:
        return {'complete': False, 'reason': 'Native self-crossing face budget exceeded'}
    tri = v[f]
    lower, upper = tri.min(1), tri.max(1)
    tol = max(float(np.ptp(v, axis=0).max()), 1.) * 1e-10
    _, ids = np.unique(np.round(v / tol).astype(np.int64), axis=0, return_inverse=True)
    ids = ids[f]
    hits, checked = [], 0
    for i in range(len(f)):
        candidates = np.flatnonzero(np.all(upper[i] + tol >= lower, axis=1)
                                     & np.all(upper + tol >= lower[i], axis=1))
        for j in candidates[candidates > i]:
            if set(ids[i]) & set(ids[j]):
                continue
            checked += 1
            if checked > max_pairs:
                return {'complete': False, 'reason': 'Native self-crossing pair budget exceeded'}
            if proper_crossing(tri[i], tri[j], tol):
                hits.append([int(i), int(j)])
    return {'complete': True, 'pairs': hits, 'count': len(hits), 'checked': checked,
            'tolerance_m': tol, 'scope': 'proper_nonadjacent_noncoplanar_only'}


def directed_edges_closed(mesh):
    """No orientation repair or deduplication; require a closed input surface."""
    v, ids = np.unique(mesh.vertices, axis=0, return_inverse=True)
    f = ids[mesh.faces]
    counts = topology(mesh)
    if any(counts[k] for k in ('boundary_edges', 'nonmanifold_edges',
                                'degenerate_faces', 'duplicate_unoriented_faces')):
        return False
    directed = Counter((int(a), int(b)) for face in f for a, b in zip(face, np.roll(face, -1)))
    return all(directed[a, b] == directed[b, a] == 1 for a, b in directed)
