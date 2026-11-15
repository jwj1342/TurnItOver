from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import trimesh

from turnitover.checkers.base import PartMesh
from turnitover.repair.native_geometry import (
    connection, convex_certificate, measure_geometry, select_pair, solid_relation,
)
from turnitover.repair.native_scopes import construction_scopes


def box(offset=0., scale=1.):
    mesh = trimesh.creation.box(extents=[scale] * 3)
    return PartMesh(np.asarray(mesh.vertices) + [offset, 0, 0], np.asarray(mesh.faces))


def scope(kind, **kwargs):
    return {'defect_id': kind, 'reference_program_sha256': 'a' * 64,
            'provenance': [{'path': 'fixture-only', 'sha256': 'b' * 64}],
            'target': {'part': 'part'}, **kwargs}


def measured(kind, a, b, **kwargs):
    histories = {role: {s: value for s in ('zero', 'mid', 'limit')}
                 for role, value in [('reference', a), ('candidate', b)]}
    return measure_geometry(kind, scope(kind, **kwargs), histories, {'reference': 'a' * 64})


@pytest.mark.parametrize('offset,relation', [(2., 'separated'), (1., 'contact'), (.8, 'overlap'), (0., 'overlap')])
def test_actual_closed_solid_relation(offset, relation):
    assert solid_relation(box(), box(offset))['relation'] == relation


def test_containment_is_overlap_even_without_surface_crossings():
    assert solid_relation(box(scale=2), box(scale=.2))['relation'] == 'overlap'


def test_open_or_nested_surfaces_are_not_solid_certificates():
    a = box()
    assert convex_certificate(PartMesh(a.vertices, a.faces[:-1])) is None
    inner = box(scale=.2)
    nested = PartMesh(np.concatenate([a.vertices, inner.vertices]),
                      np.concatenate([a.faces, inner.faces + len(a.vertices)]))
    assert convex_certificate(nested) is None
    relation = solid_relation(nested, box())
    assert relation['relation'] == 'overlap'
    assert relation['physical_certificate'] is False


def test_separating_axis_work_exhaustion_is_unknown(monkeypatch):
    import turnitover.repair.native_geometry as native
    monkeypatch.setattr(native, 'MAX_PAIRS', 1)
    assert solid_relation(box(), box(.8))['relation'] is None
    assert connection(box(), box())['connected'] is None


@pytest.mark.parametrize('kind', ['geometry.thin_sheet', 'geometry.missing_bevel'])
def test_source_feature_restoration_checks_positions_and_live_triangles(kind):
    reference = {'part': box()}
    assert measured(kind, reference, reference, tolerance_m=1e-7)['passed'] is True
    v = reference['part'].vertices.copy()
    v[:, 0] *= .02
    candidate = {'part': PartMesh(v, reference['part'].faces)}
    assert measured(kind, reference, candidate, tolerance_m=1e-7)['passed'] is False
    candidate = {'part': PartMesh(reference['part'].vertices, reference['part'].faces[:-1])}
    assert measured(kind, reference, candidate, tolerance_m=1e-7)['passed'] is False


def test_bevel_cannot_pass_by_retaining_its_box_or_unused_reference_vertices():
    # Restore all vertex records while leaving the selected native bevel faces
    # absent. Checking bounds or the vertex cloud alone would incorrectly pass.
    reference = {'part': box()}
    candidate = {'part': PartMesh(reference['part'].vertices, reference['part'].faces[:4])}
    assert measured('geometry.missing_bevel', reference, candidate, tolerance_m=1e-7)['passed'] is False


def test_face_reordering_and_unselected_part_geometry_do_not_break_component_selection():
    a = box()
    left, right = select_pair({'part': a}, {'part': PartMesh(a.vertices, a.faces[::-1])},
                             {'part': 'part', 'face_ids': [0, 1]})
    assert len(left.faces) == len(right.faces) == 2
    left, right = select_pair({'part': a}, {'part': box(scale=.5)},
                             {'part': 'part', 'vertex_ranges': [[0, len(a.vertices)]]})
    assert len(left.faces) == len(right.faces) == len(a.faces)


@pytest.mark.parametrize('kind,offset', [('structure.floating', 1.3), ('structure.interpenetration', .7)])
def test_contact_labels_use_actual_current_pair(kind, offset):
    reference = {'part': box(), 'other': box(1.)}
    support = {'part': 'other'}
    assert measured(kind, reference, reference, support=support)['passed'] is True
    candidate = {**reference, 'other': box(offset)}
    assert measured(kind, reference, candidate, support=support)['passed'] is False


def test_unhealthy_reference_overlap_cannot_authorize_a_contact_pass():
    reference = {'part': box(), 'other': box(.7)}
    assert measured('structure.interpenetration', reference, reference,
                    support={'part': 'other'})['passed'] is None


def test_contact_regression_compares_each_unchanged_part_once_per_state(monkeypatch):
    import turnitover.repair.native_geometry as native
    parts = {str(index): box(index * 2.) for index in range(6)}
    histories = {role: {'zero': parts} for role in ('reference', 'candidate')}
    original = native.same_surface
    calls = []
    def counted(a, b):
        calls.append(1)
        return original(a, b)
    monkeypatch.setattr(native, 'same_surface', counted)
    assert native.contact_regressions(histories)['passed'] is True
    assert len(calls) == len(parts)


def test_contact_regression_reuses_identical_pair_across_snapshots(monkeypatch):
    import turnitover.repair.native_geometry as native
    reference = {'part': box(), 'other': box(1.)}
    candidate = {'part': box(), 'other': box(1.3)}
    histories = {role: {state: deepcopy(parts) for state in ('zero', 'mid', 'limit')}
                 for role, parts in (('reference', reference), ('candidate', candidate))}
    original, calls = native.solid_relation, []
    def counted(a, b):
        calls.append(1)
        return original(a, b)
    monkeypatch.setattr(native, 'solid_relation', counted)
    assert native.contact_regressions(histories)['passed'] is False
    assert len(calls) == 2


def test_feature_regression_cache_is_local_and_geometry_bound(monkeypatch):
    import turnitover.repair.native_geometry as native
    histories = {role: {state: {'part': box(scale=size)} for state in ('zero', 'mid', 'limit')}
                 for role, size in (('reference', 1.), ('candidate', .8))}
    calls = []
    def matched(a, b):
        calls.append(1)
        return {'passed': False}
    monkeypatch.setattr(native, 'surface_match', matched)
    assert native.feature_regressions(histories)['passed'] is False
    assert len(calls) == 1
    histories['candidate']['limit']['part'] = box(scale=.7)
    assert native.feature_regressions(histories)['passed'] is False
    assert len(calls) == 3  # Fresh invocation; the changed snapshot gets its own key.


def test_scope_must_match_current_reference_source_and_label():
    histories = {r: {s: {'part': box()} for s in ('zero', 'mid', 'limit')}
                 for r in ('reference', 'candidate')}
    value = scope('geometry.thin_sheet', tolerance_m=0)
    assert measure_geometry('geometry.thin_sheet', value, histories,
                            {'reference': 'stale'})['passed'] is None
    value['defect_id'] = 'geometry.missing_bevel'
    assert measure_geometry('geometry.thin_sheet', value, histories,
                            {'reference': 'a' * 64})['passed'] is None


def annotation(tmp_path, report, kind):
    path = tmp_path / 'report.private.json'
    raw = json.dumps(report).encode()
    path.write_bytes(raw)
    return {'split': 'validation', 'labels': [{'defect_id': kind, 'parts': [], 'params': {}}],
            'source_roster_row': {'split': 'validation', 'report_path': str(path),
                                 'report_sha256': hashlib.sha256(raw).hexdigest()}}


def test_scope_recovery_uses_identifiers_not_historical_verdicts(tmp_path):
    row = annotation(tmp_path, {'target': 'part', 'constructed_evidence_checks_passed': True,
                               'mutation_witness': {'shortest_local_bbox_axis': 1}}, 'geometry.thin_sheet')
    before = deepcopy(row)
    recovered = construction_scopes(row, 'a' * 64)
    assert row == before
    assert recovered['labels'][0]['parts'] == ['part']
    assert recovered['scopes'][0]['target'] == {'part': 'part'}
    assert 'constructed_evidence_checks_passed' not in json.dumps(recovered)
    assert recovered['scopes'][0]['reference_program_sha256'] == 'a' * 64


@pytest.mark.parametrize('kind,fields,parts,joint', [
    ('geometry.missing_part', {'condition': {'target': 'part'}}, ['part'], None),
    ('geometry.duplicate_part', {'duplicate_target': 'part'}, ['part'], None),
    ('material.metalness', {'condition': {'part': 'part'}}, ['part'], None),
    ('material.tiling_scale', {'condition': {'target': 'part'}}, ['part'], None),
    ('kinematics.joint_type', {'joint': 'hinge', 'target_subtree': ['part']}, ['part'], 'hinge'),
    ('kinematics.joint_range', {'joint': 'hinge', 'target_subtree': ['part']}, ['part'], 'hinge'),
])
def test_original_collector_selector_spellings(tmp_path, kind, fields, parts, joint):
    row = annotation(tmp_path, {'defect_id': kind, **fields, 'historical_pass': True}, kind)
    result = construction_scopes(row, 'a' * 64)
    assert result['labels'][0]['parts'] == parts
    assert result['labels'][0].get('joint_id') == joint
    assert 'historical_pass' not in json.dumps(result)


def test_other_defect_report_cannot_supply_missing_target(tmp_path):
    row = annotation(tmp_path, {'defect_id': 'material.metalness',
        'condition': {'part': 'unrelated'}}, 'geometry.missing_part')
    assert construction_scopes(row, 'a' * 64)['labels'][0]['parts'] == []


def test_shared_report_does_not_assign_one_target_to_other_instances(tmp_path):
    row = annotation(tmp_path, {'defect_id': 'material.metalness',
        'condition': {'part': 'part'}}, 'material.metalness')
    row['labels'].append({'defect_id': 'geometry.missing_part', 'parts': []})
    recovered = construction_scopes(row, 'a' * 64)['labels']
    assert [label['parts'] for label in recovered] == [['part'], []]


def test_bevel_scope_preserves_original_small_tolerance_and_native_range(tmp_path):
    row = annotation(tmp_path, {'target': 'part', 'native_component_chamfer_proof': {
        'raw_component_chamfer_proof': {'strict_local_native_edge_proof': {
            'native_profile': {'tolerance_m': 5e-7, 'depth': .02}}},
        'lift': [{'native_vertex_start': 20, 'native_vertex_count': 40}]}}, 'geometry.missing_bevel')
    value = construction_scopes(row, 'a' * 64)['scopes'][0]
    assert value['tolerance_m'] == 5e-7
    assert value['feature_depth_m'] == .02
    assert value['target']['vertex_ranges'] == [[20, 40]]


def test_scope_recovery_checks_hash_and_rejects_test_before_reading(tmp_path):
    row = annotation(tmp_path, {'target': 'part'}, 'geometry.thin_sheet')
    row['source_roster_row']['report_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash mismatch'):
        construction_scopes(row, 'a' * 64)
    row['source_roster_row']['split'] = 'test'
    with pytest.raises(ValueError, match='Train/Validation'):
        construction_scopes(row, 'a' * 64, resolve=lambda _: pytest.fail('Read Test source'))


def test_partitioned_component_support_offsets_follow_native_collector(tmp_path):
    row = annotation(tmp_path, {'target': {'part': 'part', 'source_face_ids': [0, 2]},
        'support': {'part': 'part', 'source_face_ids': [4, 5]}}, 'structure.floating')
    value = construction_scopes(row, 'a' * 64)['scopes'][0]
    assert value['target']['part'] == 'native-contact-surface'
    assert value['support']['face_ids'] == [2, 3]


@pytest.mark.parametrize('binding', ['valid', 'candidate_mismatch', 'reference_mismatch', 'report_mismatch'])
def test_wrapper_scope_follows_only_hash_bound_original_collector(tmp_path, binding):
    wrapper = tmp_path / 'wrapper'
    collector = tmp_path / 'collector'
    wrapper.mkdir()
    collector.mkdir()
    row = annotation(wrapper, {'public_view_wrapper': True}, 'geometry.thin_sheet')
    report_path = collector / 'report.private.json'
    raw = json.dumps({'target': 'original-part', 'historical_pass': True}).encode()
    report_path.write_bytes(raw)
    reference, candidate = (collector / name for name in ('reference.ts', 'candidate.ts'))
    row['programs'] = {
        'case_reference': {'path': str(reference), 'sha256': 'a' * 64},
        'case_candidate': {'path': str(candidate), 'sha256': 'b' * 64},
    }
    hashes = {str(reference): 'a' * 64, str(candidate): 'b' * 64,
              report_path.as_posix(): hashlib.sha256(raw).hexdigest()}
    if binding == 'candidate_mismatch':
        hashes[str(candidate)] = 'c' * 64
    elif binding == 'reference_mismatch':
        hashes[str(reference)] = 'c' * 64
    elif binding == 'report_mismatch':
        hashes[report_path.as_posix()] = 'c' * 64
    (wrapper / 'preregistration.private.json').write_text(json.dumps({'inputs_sha256': hashes}))
    if binding == 'report_mismatch':
        with pytest.raises(ValueError, match='hash mismatch'):
            construction_scopes(row, 'a' * 64)
    else:
        result = construction_scopes(row, 'a' * 64)
        if binding == 'valid':
            assert result['scopes'][0]['target'] == {'part': 'original-part'}
            assert len(result['provenance']) == 3
            assert 'historical_pass' not in json.dumps(result)
        else:
            assert result['scopes'] == [None]
