"""Native triangle distances from the RP29 source checker.

Adapted pure functions only; no second renderer or saved verdict consumption.
See docs/native-scoring-provenance.json for source hashes and scope limitations.
"""
from __future__ import annotations

import numpy as np


def segment_distance(p: np.ndarray, q: np.ndarray,
                     r: np.ndarray, s: np.ndarray) -> float:
    """Shortest Euclidean distance between two closed segments."""
    u, v, w = q-p, s-r, p-r
    a, b, c, d, e = u@u, u@v, v@v, u@w, v@w
    if a <= 1e-30 and c <= 1e-30:
        return float(np.linalg.norm(p-r))
    if a <= 1e-30:
        x, y = 0., float(np.clip(e/c, 0., 1.))
    elif c <= 1e-30:
        x, y = float(np.clip(-d/a, 0., 1.)), 0.
    else:
        denom = a*c-b*b
        x = float(np.clip((b*e-c*d)/denom, 0., 1.)) if denom > 1e-30 else 0.
        y = (b*x+e)/c
        if y < 0.:
            y, x = 0., float(np.clip(-d/a, 0., 1.))
        elif y > 1.:
            y, x = 1., float(np.clip((b-d)/a, 0., 1.))
    return float(np.linalg.norm(w+x*u-y*v))


def point_triangle_distance(p: np.ndarray, t: np.ndarray) -> float:
    e0, e1 = t[1]-t[0], t[2]-t[0]
    normal = np.cross(e0, e1)
    n2 = normal@normal
    if n2 > 1e-30:
        signed = (p-t[0])@normal
        projected = p-normal*(signed/n2)
        v = projected-t[0]
        aa, ab, bb = e0@e0, e0@e1, e1@e1
        denom = aa*bb-ab*ab
        if denom > 1e-30:
            x, y = ((v@e0)*bb-(v@e1)*ab)/denom, ((v@e1)*aa-(v@e0)*ab)/denom
            if x >= 0 and y >= 0 and x+y <= 1:
                return float(abs(signed)/np.sqrt(n2))
    return min(segment_distance(p, p, t[i], t[(i+1)%3]) for i in range(3))


def plane_cut(t: np.ndarray, distances: np.ndarray, tol: float) -> np.ndarray:
    points = [t[i] for i in range(3) if abs(distances[i]) <= tol]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        if distances[i]*distances[j] < 0:
            points.append(t[i]+(t[j]-t[i])*distances[i]/(distances[i]-distances[j]))
    return np.asarray(points)


def crossing(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """Proper noncoplanar surface crossing, excluding boundary-only contacts."""
    na, nb = np.cross(a[1]-a[0], a[2]-a[0]), np.cross(b[1]-b[0], b[2]-b[0])
    if min(np.linalg.norm(na), np.linalg.norm(nb)) <= tol*tol:
        return False
    na, nb = na/np.linalg.norm(na), nb/np.linalg.norm(nb)
    da, db = (a-b[0])@nb, (b-a[0])@na
    # Require strict straddling in both triangles, not tangency.
    if not (da.min() < -tol and da.max() > tol and db.min() < -tol and db.max() > tol):
        return False
    line = np.cross(na, nb)
    if np.linalg.norm(line) <= 1e-10:
        return False
    line /= np.linalg.norm(line)
    pa, pb = plane_cut(a, da, tol), plane_cut(b, db, tol)
    if min(len(pa), len(pb)) < 2:
        return False
    ia, ib = pa@line, pb@line
    return bool(min(ia.max(), ib.max())-max(ia.min(), ib.min()) > 4*tol)


def triangle_distance(a: np.ndarray, b: np.ndarray, tol: float) -> float:
    if crossing(a, b, tol):
        return 0.
    # Include edge/face piercings at triangle boundaries, which the deliberately
    # strict crossing counter excludes. Vertex/edge distances alone miss them.
    for edge_triangle, face in ((a, b), (b, a)):
        normal = np.cross(face[1]-face[0], face[2]-face[0])
        if np.linalg.norm(normal) <= 1e-30:
            continue
        for i in range(3):
            p, q = edge_triangle[i], edge_triangle[(i+1)%3]
            dp, dq = (p-face[0])@normal, (q-face[0])@normal
            if dp*dq <= 0 and abs(dp-dq) > 1e-30:
                hit = p+(q-p)*(dp/(dp-dq))
                if point_triangle_distance(hit, face) <= 1e-12:
                    return 0.
    # These features exhaust distances between disjoint closed triangles.
    return min(*(point_triangle_distance(p, b) for p in a),
               *(point_triangle_distance(p, a) for p in b),
               *(segment_distance(a[i], a[(i+1)%3], b[j], b[(j+1)%3])
                 for i in range(3) for j in range(3)))


def topology(vertices: np.ndarray, faces: np.ndarray, tol: float) -> dict:
    _, inverse = np.unique(np.round(vertices/tol).astype(np.int64), axis=0, return_inverse=True)
    welded = inverse[faces]
    degenerate = np.any(np.diff(np.sort(welded, axis=1), axis=1) == 0, axis=1)
    valid = welded[~degenerate]
    unique = np.unique(np.sort(valid, axis=1), axis=0)
    edges = np.sort(np.concatenate((unique[:, [0, 1]], unique[:, [1, 2]], unique[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {"vertex_count": len(vertices), "triangle_count": len(faces),
            "weld_tolerance": tol, "degenerate_by_welding": int(degenerate.sum()),
            "duplicate_faces_ignoring_winding": int(len(valid)-len(unique)),
            "unique_face_boundary_edges": int((counts == 1).sum()),
            "unique_face_nonmanifold_edges": int((counts > 2).sum()),
            "watertight_certified": False}


def surface_check(a: np.ndarray, b: np.ndarray, tol: float) -> dict:
    alo, ahi, blo, bhi = a.min(1), a.max(1), b.min(1), b.max(1)
    delta = np.maximum(0., np.maximum(alo[:, None]-bhi, blo-ahi[:, None]))
    lower = np.linalg.norm(delta, axis=2)
    intersecting, coplanar_close = [], []
    for i, j in np.argwhere(lower <= tol):
        aa, bb = a[i], b[j]
        if crossing(aa, bb, tol):
            intersecting.append([int(i), int(j)])
        else:
            na, nb = np.cross(aa[1]-aa[0], aa[2]-aa[0]), np.cross(bb[1]-bb[0], bb[2]-bb[0])
            if min(np.linalg.norm(na), np.linalg.norm(nb)) > tol*tol:
                na, nb = na/np.linalg.norm(na), nb/np.linalg.norm(nb)
                if np.linalg.norm(np.cross(na, nb)) < 1e-8 and abs((bb[0]-aa[0])@na) <= tol:
                    if triangle_distance(aa, bb, tol) <= tol:
                        coplanar_close.append([int(i), int(j)])
    best, witness, tested = float("inf"), None, 0
    for flat in np.argsort(lower, axis=None):
        i, j = np.unravel_index(flat, lower.shape)
        if lower[i, j] > best+tol:
            break
        distance = triangle_distance(a[i], b[j], tol)
        tested += 1
        if distance < best:
            best, witness = distance, [int(i), int(j)]
        if best <= tol:
            break
    return {"proper_noncoplanar_crossing_count": len(intersecting),
            "proper_noncoplanar_crossing_pairs": intersecting,
            "coplanar_overlap_or_contact_count": len(coplanar_close),
            "minimum_surface_distance": best, "minimum_distance_triangle_pair": witness,
            "distance_pairs_evaluated": tested,
            "surface_separated_over_tolerance": bool(best > tol),
            "solid_overlap_or_support_certified": False}
