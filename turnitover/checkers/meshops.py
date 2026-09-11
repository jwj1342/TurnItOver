"""Pure mesh math on numpy arrays. No browser, fully unit-testable."""
from __future__ import annotations

import numpy as np
import trimesh.triangles
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


def point_surface_distance(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, k: int = 16) -> np.ndarray:
    """Distance from each point to the triangle surface.

    Exact against the k triangles whose centroids are nearest to the point (all triangles when
    the mesh has <= k). Tessellation-independent, unlike point-to-point Chamfer.
    """
    tris = vertices[faces].astype(np.float64)
    k = min(k, len(tris))
    _, idx = cKDTree(tris.mean(axis=1)).query(points, k=k)
    idx = np.asarray(idx).reshape(len(points), k)
    q = np.repeat(points.astype(np.float64), k, axis=0)
    closest = trimesh.triangles.closest_point(tris[idx.ravel()], q)
    return np.linalg.norm(closest - q, axis=1).reshape(len(points), k).min(axis=1)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    return float(0.5 * (da.mean() + db.mean()))


def surface_chamfer(va: np.ndarray, fa: np.ndarray, vb: np.ndarray, fb: np.ndarray, n: int, seed: int = 0) -> float:
    """Symmetric mean point-to-surface distance between two triangle meshes."""
    pa = sample_surface(va, fa, n, np.random.default_rng(seed))
    pb = sample_surface(vb, fb, n, np.random.default_rng(seed))
    return float(0.5 * (point_surface_distance(pa, vb, fb).mean() + point_surface_distance(pb, va, fa).mean()))
