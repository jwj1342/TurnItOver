import numpy as np
import trimesh

from turnitover.checkers import meshops


def _box(offset=(0, 0, 0), extents=(1, 1, 1)):
    m = trimesh.creation.box(extents=extents)
    return np.asarray(m.vertices, dtype=np.float32) + np.asarray(offset, dtype=np.float32), np.asarray(m.faces, dtype=np.uint32)


def _subdivided_box(n=4):
    m = trimesh.creation.box(extents=(1, 1, 1))
    for _ in range(n):
        m = m.subdivide()
    return np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.uint32)


def test_iou_identity_and_disjoint():
    v, _ = _box()
    assert meshops.aabb_iou(v, v) == 1.0
    v2, _ = _box(offset=(5, 0, 0))
    assert meshops.aabb_iou(v, v2) == 0.0


def test_sample_surface_on_surface():
    v, f = _box()
    pts = meshops.sample_surface(v, f, 500, np.random.default_rng(1))
    assert pts.shape == (500, 3)
    assert np.all(np.abs(pts).max(axis=1) > 0.5 - 1e-5)  # every sample lies on a face of the unit cube


def test_point_surface_distance_exact():
    v, f = _box()
    d = meshops.point_surface_distance(np.array([[0, 0, 0.7], [0, 0, 0], [2, 2, 2]], dtype=np.float32), v, f)
    assert np.allclose(d, [0.2, 0.5, np.sqrt(3) * 1.5], atol=1e-6)


def test_surface_chamfer_tessellation_invariant():
    v, f = _box()
    vs, fs = _subdivided_box()
    assert meshops.surface_chamfer(v, f, v, f, 2000) < 1e-12
    assert meshops.surface_chamfer(v, f, vs, fs, 2000) < 1e-5  # same surface, different triangles
    v2, f2 = _box(offset=(0.3, 0, 0))
    d = meshops.surface_chamfer(v, f, v2, f2, 2000)
    assert 0.05 < d < 0.3


def test_large_triangle_not_hidden_by_nearer_centroids():
    # The query lies on a large triangle, while 20 tiny triangles have nearer centroids.
    large = np.array([[-100., -100., 0], [100., -100., 0], [0, 100., 0]])
    tiny = [np.array([[0., 0., z], [.01, 0., z], [0., .01, z]]) for z in np.linspace(.1, .2, 20)]
    vertices = np.concatenate([large, *tiny])
    faces = np.arange(len(vertices)).reshape(-1, 3)
    assert meshops.point_surface_distance(np.array([[0., 0., 0.]]), vertices, faces)[0] < 1e-12


def test_degenerate_faces_do_not_produce_nan():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 2]])
    distances = meshops.point_surface_distance(np.array([[.2, .2, .3], [.5, -.2, 0]]), vertices, faces)
    assert np.allclose(distances, [.3, .2])


def test_bounded_search_matches_brute_force():
    rng = np.random.default_rng(92)
    vertices = rng.normal(size=(120, 3))
    faces = np.arange(120).reshape(-1, 3)
    points = rng.normal(size=(30, 3))
    triangles = vertices[faces]
    paired = trimesh.triangles.closest_point(np.tile(triangles, (len(points), 1, 1)), np.repeat(points, len(faces), axis=0))
    brute = np.linalg.norm(paired-np.repeat(points, len(faces), axis=0), axis=1).reshape(len(points), -1).min(axis=1)
    assert np.allclose(meshops.point_surface_distance(points, vertices, faces), brute, atol=1e-12)
