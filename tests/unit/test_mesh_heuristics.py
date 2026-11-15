"""Reference-relative scoring accepts remeshing but rejects geometric shortcuts."""
import numpy as np
import pytest
import trimesh

from turnitover.checkers.base import PartMesh
from turnitover.checkers.mesh_heuristics import (
    HEURISTICS, component_count, feature_match, same_surface, sampled_relation, surface_match,
)
from turnitover.checkers.meshops import point_surface_distance, sample_surface
from turnitover.repair.native_geometry import contact_regressions, feature_regressions, measure_geometry, select_pair


def part(mesh):
    return PartMesh(np.asarray(mesh.vertices).copy(), np.asarray(mesh.faces).copy())


def box():
    return part(trimesh.creation.box())


def history(a, b):
    return {role: {s: {'part': mesh} for s in ('zero', 'mid', 'limit')}
            for role, mesh in [('reference', a), ('candidate', b)]}


def test_nearest_triangle_ids_correspond_to_exact_surface_distances():
    mesh = box()
    distance, ids = point_surface_distance(np.array([[2., 0, 0], [0, 0, 0.]]),
        mesh.vertices, mesh.faces, return_face_ids=True)
    assert distance == pytest.approx([1.5, .5])
    assert np.all((ids >= 0) & (ids < len(mesh.faces)))
    assert np.all(mesh.vertices[mesh.faces[ids[0]]][:, 0] == .5)


def test_degenerate_surface_sampling_reports_unavailable_face_ids():
    points, ids = sample_surface(np.zeros((3, 3)), np.array([[0, 1, 2]]), 5,
                                np.random.default_rng(0), return_face_ids=True)
    assert points.shape == (5, 3) and np.all(ids == -1)


@pytest.mark.parametrize('kind', ['geometry.thin_sheet', 'geometry.missing_bevel'])
def test_feature_repair_accepts_actual_subdivision(kind):
    mesh = trimesh.creation.box()
    a, b = part(mesh), part(mesh.subdivide())
    assert len(a.vertices) != len(b.vertices)
    assert feature_match(a, b, kind, tolerance=1e-7)['passed'] is True
    assert feature_regressions(history(a, b))['passed'] is True


def test_triangle_position_equivalence_accepts_numbering_and_face_order_changes():
    a = box()
    order = np.arange(len(a.vertices))[::-1]
    inverse = np.argsort(order)
    b = PartMesh(a.vertices[order], inverse[a.faces[::-1]])
    assert same_surface(a, b)
    assert surface_match(a, b)['passed'] is True


@pytest.mark.parametrize('factor', [.02, 2.])
def test_thickness_rejects_flattening_and_overinflation(factor):
    a = box()
    b = PartMesh(a.vertices * [factor, 1, 1], a.faces)
    assert feature_match(a, b, 'geometry.thin_sheet')['passed'] is False


def test_unused_bbox_vertices_do_not_restore_thickness():
    a = box()
    vertices = np.concatenate([a.vertices * [.01, 1, 1], a.vertices])
    b = PartMesh(vertices, a.faces)
    assert feature_match(a, b, 'geometry.thin_sheet')['passed'] is False


def test_local_bevel_profile_detects_lost_chamfer_and_accepts_its_remesh():
    # A box with corners clipped by 0.1; its axis-aligned bounds match a cube.
    points = np.array([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    clipped = []
    for p in points:
        for axis in range(3):
            q = p.copy()
            q[axis] *= .8
            clipped.append(q)
    bevel = trimesh.convex.convex_hull(np.array(clipped))
    reference = part(bevel)
    assert feature_match(reference, box(), 'geometry.missing_bevel', depth=.1)['passed'] is False
    assert feature_match(reference, part(bevel.subdivide()), 'geometry.missing_bevel', depth=.1)['passed'] is True


def test_component_selector_recovers_a_remeshed_component_in_a_larger_part():
    target = trimesh.creation.box()
    other = trimesh.creation.box(transform=trimesh.transformations.translation_matrix([3., 0, 0]))
    original = part(trimesh.util.concatenate([target, other]))
    remeshed = part(trimesh.util.concatenate([target.subdivide(), other.subdivide()]))
    a, b = select_pair({'part': original}, {'part': remeshed}, {'part': 'part', 'face_ids': list(range(12))})
    assert surface_match(a, b)['passed'] is True


def nonconvex():
    # Connected L-shaped triangular prism, without Boolean engine dependencies.
    polygon = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    cap = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]])
    vertices = np.vstack([np.column_stack([polygon, np.zeros(6)]), np.column_stack([polygon, np.ones(6)])])
    faces = [*cap[:, ::-1], *(cap + 6)]
    for i in range(6):
        j = (i + 1) % 6
        faces.extend([[i, j, j + 6], [i, j + 6, i + 6]])
    return PartMesh(vertices, np.array(faces))


@pytest.mark.parametrize('offset,relation', [([.5, .5, .5], 'overlap'),
                                            ([1.5, 1.5, .5], 'separated')])
def test_nonconvex_occupancy_handles_containment_and_empty_concavity(offset, relation):
    b = part(trimesh.creation.box(extents=[.2] * 3,
        transform=trimesh.transformations.translation_matrix(offset)))
    result = sampled_relation(nonconvex(), b)
    assert result['relation'] == relation, result
    assert result['physical_certificate'] is False


def test_sampled_coincident_nonconvex_objects_overlap_and_work_exhaustion_is_unknown():
    a = nonconvex()
    assert sampled_relation(a, a)['relation'] == 'overlap'
    assert sampled_relation(a, a, max_pairs=1)['relation'] is None


def test_shifted_duplicate_is_not_hidden_by_noncoincident_faces():
    a = trimesh.creation.box()
    duplicate = a.copy()
    duplicate.apply_translation([.02, 0, 0])
    assert component_count(part(a)) == 1
    assert component_count(part(trimesh.util.concatenate([a, duplicate]))) == 2


def test_reference_assembly_intersections_do_not_make_unchanged_regressions_unknown():
    a = box()
    b = PartMesh(a.vertices + [.3, 0, 0], a.faces)
    states = {s: {'part': a, 'other': b} for s in ('zero', 'mid', 'limit')}
    assert contact_regressions({'reference': states, 'candidate': states})['passed'] is True


@pytest.mark.parametrize('offset,passed', [(.5, False), (.9, True), (1., True), (1.2, False)])
def test_baseline_overlap_cannot_hide_worsened_penetration_or_disconnection(offset, passed):
    a = box()
    reference = {'part': a, 'other': PartMesh(a.vertices + [.8, 0, 0], a.faces)}
    candidate = {'part': a, 'other': PartMesh(a.vertices + [offset, 0, 0], a.faces)}
    result = contact_regressions({'reference': {'zero': reference}, 'candidate': {'zero': candidate}})
    assert result['passed'] is passed


def test_floating_rejects_a_needle_in_place_of_a_contact_patch():
    a = trimesh.creation.box()
    support = trimesh.creation.box(transform=trimesh.transformations.translation_matrix([0, 0, -1.]))
    raised = a.copy()
    raised.apply_translation([0, 0, .2])
    needle = trimesh.creation.box(extents=[.01, .01, .2],
        transform=trimesh.transformations.translation_matrix([0, 0, -.4]))
    damaged = part(trimesh.util.concatenate([raised, needle]))
    histories = {role: {s: {'part': mesh, 'support': part(support)} for s in ('zero', 'mid', 'limit')}
        for role, mesh in [('reference', part(a)), ('candidate', damaged)]}
    scope = {'defect_id': 'structure.floating', 'reference_program_sha256': 'a' * 64,
        'provenance': [{'path': 'analytic', 'sha256': 'b' * 64}],
        'target': {'part': 'part'}, 'support': {'part': 'support'}}
    result = measure_geometry('structure.floating', scope, histories, {'reference': 'a' * 64})
    assert result['passed'] is False
