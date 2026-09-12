"""Pure mesh math on numpy arrays. No browser, fully unit-testable."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def aabb(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return vertices.min(axis=0), vertices.max(axis=0)


def aabb_iou(a: np.ndarray, b: np.ndarray) -> float:
    amin, amax = aabb(a)
    bmin, bmax = aabb(b)
    inter = np.clip(np.minimum(amax, bmax) - np.maximum(amin, bmin), 0.0, None).prod()
    va = np.clip(amax - amin, 0.0, None).prod()
    vb = np.clip(bmax - bmin, 0.0, None).prod()
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0


def sample_surface(vertices: np.ndarray, faces: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Area-weighted uniform surface samples (deterministic given rng)."""
    tri = vertices[faces]  # (M, 3, 3)
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if areas.sum() <= 0:
        return vertices[rng.integers(0, len(vertices), size=n)]
    idx = rng.choice(len(faces), size=n, p=areas / areas.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w0, w1, w2 = 1 - s, s * (1 - r2), s * r2
    t = tri[idx]
    return w0[:, None] * t[:, 0] + w1[:, None] * t[:, 1] + w2[:, None] * t[:, 2]


def _point_triangle_distance(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Vectorized paired distances, including degenerate triangles as segments/points."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    best = np.full(len(points), np.inf)
    normal = np.cross(b-a, c-a)
    norm2 = np.einsum("ij,ij->i", normal, normal)
    inside = norm2 > 0
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end-start
        delta = points-start
        denom = np.einsum("ij,ij->i", edge, edge)
        t = np.divide(np.einsum("ij,ij->i", delta, edge), denom,
                      out=np.zeros_like(denom), where=denom > 0)
        residual = delta - np.clip(t, 0, 1)[:, None] * edge
        best = np.minimum(best, np.einsum("ij,ij->i", residual, residual))
        inside &= np.einsum("ij,ij->i", np.cross(edge, delta), normal) >= 0
    height = np.einsum("ij,ij->i", points-a, normal)
    plane = np.divide(height*height, norm2, out=np.full_like(norm2, np.inf), where=norm2 > 0)
    best = np.where(inside, np.minimum(best, plane), best)
    return np.sqrt(np.maximum(best, 0))


def point_surface_distance(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, k: int = 16) -> np.ndarray:
    """Exact nearest triangle distance with conservative centroid-ball pruning.

    The nearest referenced vertex supplies an upper bound. A triangle can improve
    that bound only when its centroid ball reaches the point's upper-bound ball.
    Unlike fixed-k nearest-centroid lookup, this cannot omit a large nearby face.
    ``k`` remains accepted for source compatibility; it no longer limits candidates.
    """
    triangles = vertices[faces].astype(np.float64)
    points = np.asarray(points, dtype=np.float64)
    if not len(triangles):
        raise ValueError("Point-to-surface distance requires triangles")
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles-centers[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(centers)
    upper = cKDTree(vertices[np.unique(faces)]).query(points)[0]
    epsilon = 1e-12 * max(1., float(np.abs(vertices).max()), float(np.abs(points).max()) if len(points) else 1.)
    result = np.full(len(points), np.inf)
    for start in range(0, len(points), 64):
        chunk = points[start:start+64]
        neighbours = tree.query_ball_point(chunk, upper[start:start+64] + radii.max() + epsilon)
        point_ids = np.repeat(np.arange(len(chunk)), [len(x) for x in neighbours])
        tri_ids = np.concatenate(neighbours).astype(int)
        bound = np.linalg.norm(centers[tri_ids]-chunk[point_ids], axis=1)-radii[tri_ids]
        keep = bound <= upper[start+point_ids] + epsilon
        point_ids, tri_ids = point_ids[keep], tri_ids[keep]
        distances = _point_triangle_distance(chunk[point_ids], triangles[tri_ids])
        np.minimum.at(result, start+point_ids, distances)
    return result


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    return float(0.5 * (da.mean() + db.mean()))


def surface_chamfer(va: np.ndarray, fa: np.ndarray, vb: np.ndarray, fb: np.ndarray, n: int, seed: int = 0) -> float:
    """Symmetric mean point-to-surface distance between two triangle meshes."""
    if np.array_equal(va, vb) and np.array_equal(fa, fb):
        return 0.0
    pa = sample_surface(va, fa, n, np.random.default_rng(seed))
    pb = sample_surface(vb, fb, n, np.random.default_rng(seed))
    return float(0.5 * (point_surface_distance(pa, vb, fb).mean() + point_surface_distance(pb, va, fa).mean()))
