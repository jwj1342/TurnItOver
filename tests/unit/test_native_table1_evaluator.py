import base64
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rp29_native_replay import run, save_private_evidence, state_commands
from turnitover.repair.native_evaluator import (
    ALIGNMENT_DEFECTS, SCORING_CONTRACT, make_audit, make_gold, measure_replay,
)
from turnitover.verifier.gt_validator import _check_contract
from turnitover.repair.native_properties import PROPERTIES, TEXTURES
from turnitover.checkers.native_mesh import self_crossings


def test_scoring_environment_exercises_lazy_normal_repair_dependency(monkeypatch):
    from importlib import metadata
    from turnitover.repair.native_evaluator import check_environment
    report = check_environment()
    assert report['ready'] is True and report['normal_repair_smoke'] is True
    original_version = metadata.version
    def without_networkx(name):
        if name == 'networkx':
            raise metadata.PackageNotFoundError(name)
        return original_version(name)
    monkeypatch.setattr(metadata, 'version', without_networkx)
    report = check_environment()
    assert report['ready'] is False and report['missing'] == ['networkx']


def test_topology_cache_reuses_only_byte_identical_meshes(native, monkeypatch):
    import turnitover.repair.native_evaluator as evaluator
    _, docs, _ = native
    calls = []
    original = evaluator.self_crossings
    def measured(mesh):
        calls.append(mesh.vertices.copy())
        return original(mesh)
    monkeypatch.setattr(evaluator, 'self_crossings', measured)
    first = evaluator._topology(docs)
    assert len(calls) == 2
    docs['candidate']['states']['mid']['geometry']['parts'][0] = mesh(offset=.2)
    calls.clear()
    second = evaluator._topology(docs)
    assert len(calls) == 3
    assert first['zero'] == second['zero']


def mesh(name='part', offset=0):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='<f4') + offset
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype='<u4')
    return {'name': name, 'vertex_count': 4, 'triangle_count': 4,
            'positions_b64': base64.b64encode(vertices.astype('<f4').tobytes()).decode(),
            'indices_b64': base64.b64encode(faces.tobytes()).decode()}


@pytest.fixture
def native(tmp_path):
    source = 'current program'
    digest = hashlib.sha256(source.encode()).hexdigest()
    doc = {'schema_version': 1, 'contract': {'parts': ['part', 'other'],
        'joints': {'joint': {'type': 'revolute', 'part': 'part', 'limits': [0, 1]}}},
        'states': {state: {'geometry': {'parts': [mesh(), mesh('other', 3)]}}
                   for state in ('zero', 'mid', 'limit')}}
    docs = {role: {**deepcopy(doc), 'role': role, 'program_sha256': digest}
            for role in ('reference', 'candidate')}

    def evidence():
        private = {role: save_private_evidence(tmp_path, role, doc) for role, doc in docs.items()}
        return {'status': 'complete', 'gate': {'passed': True},
                'program_sha256': {'reference': digest, 'candidate': digest},
                'private_evidence': private,
                'runtime': [{'role': 'candidate', 'state': state, 'kind': 'public_stats',
                             'value': {'triangles_scene': 8, 'draw_calls': 2}}
                            for state in ('zero', 'mid', 'limit')]}
    return source, docs, evidence


def audit_case(kind='structure.part_offset'):
    return {'private_labels': [{'defect_id': kind, 'parts': ['part']}],
            'public_context': {'max_triangles': 100, 'max_draw_calls': 10}}


@pytest.mark.parametrize('kind', ALIGNMENT_DEFECTS)
def test_existing_alignment_measures_repaired_and_corrupt_targets(native, kind):
    source, docs, replay = native
    audit = make_audit(audit_case(kind), lambda _: replay())
    repaired = audit(source)
    assert repaired['defects'][0]['resolved'] is True
    # Missing private readbacks/scopes cannot become a vacuous full29 pass.
    assert repaired['passed'] is None and repaired['table1_eligible'] is False
    assert repaired['program_checks']['geometry.alignment'] is True
    docs['candidate']['states']['mid']['geometry']['parts'][0] = mesh(offset=.2)
    broken = audit(source)
    assert broken['defects'][0]['resolved'] is False
    assert broken['passed'] is False


def test_new_regression_fails_even_when_original_label_resolved(native):
    source, docs, replay = native
    # Introduce a different-part regression after fixing the labelled part.
    docs['candidate']['states']['limit']['geometry']['parts'][1] = mesh('other', 8)
    result = make_audit(audit_case(), lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is True
    assert result['program_checks']['geometry.alignment'] is False
    assert result['passed'] is False


def test_global_scale_predicate_covers_all_reference_parts(native):
    source, docs, replay = native
    docs['candidate']['states']['mid']['geometry']['parts'][1] = mesh('other', 8)
    result = make_audit(audit_case('geometry.global_scale'), lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is False


@pytest.mark.parametrize('kind', ['geometry.global_scale', 'geometry.anisotropic_scale'])
def test_global_alignment_ignores_empty_abi_container_but_checks_history(native, kind):
    source, docs, replay = native
    for doc in docs.values():
        doc['contract']['parts'].append('base')
        doc['return_to_zero'] = deepcopy(doc['states']['zero'])
    case = audit_case(kind)
    case['private_labels'][0]['parts'] = []
    audit = make_audit(case, lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    docs['candidate']['return_to_zero']['geometry']['parts'][0] = mesh(offset=.2)
    result = audit(source)
    assert result['defects'][0]['resolved'] is False
    assert result['program_checks']['geometry.alignment'] is False


def test_missing_part_in_authored_history_is_not_hidden_by_three_states(native):
    source, docs, replay = native
    for doc in docs.values():
        doc['authored_history'] = {'complete': True, 'snapshots': [deepcopy(doc['states']['zero'])]}
    audit = make_audit(audit_case('geometry.missing_part'), lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    docs['candidate']['authored_history']['snapshots'][0]['geometry']['parts'] = [mesh('other', 3)]
    assert audit(source)['defects'][0]['resolved'] is False


def test_target_duplicate_under_another_abi_owner_is_not_resolved(native):
    source, docs, replay = native
    for doc in docs.values():
        doc['return_to_zero'] = deepcopy(doc['states']['zero'])
    audit = make_audit(audit_case('geometry.duplicate_part'), lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    docs['candidate']['return_to_zero']['geometry']['parts'].append(mesh('different-owner'))
    result = audit(source)
    assert result['defects'][0]['resolved'] is False
    assert result['measurements'][0]['cross_owner_target_face_multiplicity'] is False


def test_duplicate_predicate_does_not_borrow_unrelated_part_failure(native):
    source, docs, replay = native
    docs['candidate']['states']['mid']['geometry']['parts'].append(mesh('other-clone', 3))
    result = make_audit(audit_case('geometry.duplicate_part'), lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is True
    assert result['program_checks']['geometry.alignment'] is False


def test_unmatched_history_cannot_pass_alignment_or_presence(native):
    source, docs, replay = native
    docs['reference']['return_to_zero'] = deepcopy(docs['reference']['states']['zero'])
    result = make_audit(audit_case('geometry.missing_part'), lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is None
    assert result['program_checks']['geometry.alignment'] is None


@pytest.mark.parametrize('kind', ['kinematics.joint_axis', 'kinematics.joint_type', 'kinematics.joint_range'])
def test_native_joint_parameter_alias_selects_only_named_joint(native, kind):
    source, docs, replay = native
    case = audit_case(kind)
    case['private_labels'][0].update(parts=[], params={'joint': 'joint'})
    audit = make_audit(case, lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    if kind == 'kinematics.joint_axis':
        docs['candidate']['states']['mid']['geometry']['parts'][0] = mesh(offset=.2)
    else:
        field, value = ('type', 'prismatic') if kind.endswith('type') else ('limits', [0, .5])
        docs['candidate']['contract']['joints']['joint'][field] = value
    assert audit(source)['defects'][0]['resolved'] is False
    case['private_labels'][0]['params']['joint'] = 'unknown'
    assert audit(source)['defects'][0]['resolved'] is None


@pytest.mark.parametrize('field,value,kind', [
    ('type', 'prismatic', 'kinematics.joint_type'),
    ('limits', [0, .5], 'kinematics.joint_range'),
])
def test_joint_metadata_cannot_hide_behind_identical_rendered_geometry(native, field, value, kind):
    source, docs, replay = native
    audit = make_audit(audit_case(kind), lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    docs['candidate']['contract']['joints']['joint'][field] = value
    result = audit(source)
    assert result['defects'][0]['resolved'] is False
    assert result['program_checks']['program.contract'] is False


@pytest.mark.parametrize('kind', ['kinematics.joint_type', 'kinematics.joint_range'])
def test_joint_metadata_resolution_is_independent_of_same_part_geometry_defect(native, kind):
    source, docs, replay = native
    case = audit_case(kind)
    case['private_labels'].append({'defect_id': 'structure.part_offset', 'parts': ['part']})
    docs['candidate']['states']['mid']['geometry']['parts'][0] = mesh(offset=.2)
    result = make_audit(case, lambda _: replay())(source)
    assert [row['resolved'] for row in result['defects']] == [True, False]
    assert result['passed'] is False


def test_missing_export_is_unknown_but_measured_missing_part_is_failure(native):
    source, docs, replay = native
    audit = make_audit(audit_case('geometry.missing_part'), lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    del docs['candidate']['states']['limit']
    assert audit(source)['defects'][0]['resolved'] is None
    docs['candidate']['states']['limit'] = {'geometry': {'parts': []}}
    result = audit(source)
    assert result['defects'][0]['resolved'] is False
    assert result['passed'] is False


@pytest.mark.parametrize('change', ['empty_reference', 'bad_indices', 'nan_vertices', 'missing_state'])
def test_incomplete_geometry_never_becomes_permissive_alignment(native, change):
    source, docs, replay = native
    if change == 'empty_reference':
        docs['reference']['states']['mid']['geometry']['parts'] = []
    elif change == 'bad_indices':
        docs['candidate']['states']['mid']['geometry']['parts'][0]['indices_b64'] = base64.b64encode(
            np.array([[0, 1, 999]], dtype='<u4').tobytes()).decode()
        docs['candidate']['states']['mid']['geometry']['parts'][0]['triangle_count'] = 1
    elif change == 'nan_vertices':
        docs['candidate']['states']['mid']['geometry']['parts'][0]['positions_b64'] = base64.b64encode(
            np.full((4, 3), np.nan, dtype='<f4').tobytes()).decode()
    else:
        del docs['candidate']['states']['mid']
    result = make_audit(audit_case(), lambda _: replay())(source)
    assert result['program_checks']['geometry.alignment'] is None
    assert result['defects'][0]['resolved'] is None
    assert result['passed'] is not True


def test_unknown_label_and_multidefect_instances_are_retained(native):
    source, docs, replay = native
    case = audit_case()
    case['private_labels'] += [{'defect_id': 'geometry.missing_bevel', 'parts': ['part']},
                               {'defect_id': 'structure.part_offset', 'parts': ['other']}]
    docs['candidate']['states']['mid']['geometry']['parts'][1] = mesh('other', 8)
    result = make_audit(case, lambda _: replay())(source)
    assert [row['resolved'] for row in result['defects']] == [True, None, False]
    assert result['passed'] is False
    assert 'geometry.missing_bevel' in result['checker_coverage']['supported']
    assert result['measurements'][1]['reason'].startswith('Missing hash-bound construction scope')


def test_scoring_checks_private_bytes_source_and_role(native):
    source, docs, replay = native
    value = replay()
    path = value['private_evidence']['candidate']['path']
    from pathlib import Path
    Path(path).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        make_audit(audit_case(), lambda _: value)(source)
    value = replay()
    value['private_evidence']['candidate'] = value['private_evidence']['reference']
    with pytest.raises(ValueError, match='role/source'):
        make_audit(audit_case(), lambda _: value)(source)
    with pytest.raises(ValueError, match='current-source'):
        make_audit(audit_case(), lambda _: value)('stale')


def test_compile_failure_is_scored_without_private_exports_and_can_be_repaired(native):
    source, _, replay = native
    bad = {'status': 'complete', 'gate': {'passed': False}, 'program_sha256':
           {'candidate': hashlib.sha256(b'broken').hexdigest()}}
    audit = make_audit(audit_case(), lambda text: bad if text == 'broken' else replay())
    assert audit('broken')['passed'] is False
    assert audit('broken')['defects'][0]['resolved'] is None
    assert audit(source)['defects'][0]['resolved'] is True


def test_budget_defects_and_regression_use_explicit_caps(native):
    source, _, replay = native
    value = replay()
    case = audit_case('runtime.draw_calls')
    value['runtime'][1]['value']['draw_calls'] = 11
    result = make_audit(case, lambda _: value)(source)
    assert result['defects'][0]['resolved'] is False
    assert result['passed'] is False
    case['public_context'] = {}
    result = make_audit(case, lambda _: value)(source)
    assert result['defects'][0]['resolved'] is None
    assert result['program_checks']['runtime.budget'] is None


def test_contract_checker_does_not_truncate_different_limit_lengths():
    reference = {'parts': ['part'], 'joints': {'joint':
        {'type': 'revolute', 'part': 'part', 'limits': [0, 1]}}}
    candidate = deepcopy(reference)
    candidate['joints']['joint']['limits'] = [0]
    assert _check_contract(candidate, reference)['passed'] is False


@pytest.mark.parametrize('kind,key,stats_key', [
    ('runtime.draw_calls', 'max_draw_calls', 'draw_calls'),
    ('runtime.triangle_budget', 'max_triangles', 'triangles_scene'),
])
def test_public_task_budget_overrides_context_sentinel_independently(native, kind, key, stats_key):
    source, _, replay = native
    case = audit_case(kind)
    case['public_context'] = {'max_triangles': 2_147_483_647, 'max_draw_calls': 2_147_483_647}
    case['task_runtime_requirements'] = [{'payload': {
        'measurement_kind': 'public_task_budget', 'observation': {'task_' + key: 6}}}]
    value = replay()
    value['runtime'][1]['value'][stats_key] = 872
    result = make_audit(case, lambda _: value)(source)
    assert result['defects'][0]['resolved'] is False
    assert result['program_checks']['runtime.budget'] is False
    assert result['passed'] is False
    assert result['measurements'][0]['caps'][key] == 6
    for row in value['runtime']:
        row['value'][stats_key] = 3
    assert make_audit(case, lambda _: value)(source)['defects'][0]['resolved'] is True
    assert case['public_context'][key] == 2_147_483_647


def test_context_sentinels_and_unmarked_runtime_are_not_public_caps(native):
    source, _, replay = native
    case = audit_case('runtime.draw_calls')
    case['public_context'] = {'max_triangles': 2_147_483_647, 'max_draw_calls': 2_147_483_647}
    case['task_runtime_requirements'] = [{'payload': {'observation': {'task_max_draw_calls': 6}}}]
    result = make_audit(case, lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is None
    assert result['program_checks']['runtime.budget'] is None


def test_conflicting_public_task_caps_are_rejected(native):
    _, _, replay = native
    case = audit_case('runtime.draw_calls')
    case['task_runtime_requirements'] = [{'payload': {
        'measurement_kind': 'public_task_budget', 'observation': {'task_max_draw_calls': 6}}}]
    with pytest.raises(ValueError, match='Conflicting explicit public runtime caps'):
        make_audit(case, lambda _: replay())


def test_range_repair_uses_current_detents_while_legacy_numeric_policy_is_preserved():
    case = {'commands': [[state, {'joint': value}] for state, value in
                         zip(('zero', 'mid', 'limit'), (0, 1, 2))],
            'commands_by_role': {'candidate': [[state, {'joint': value}] for state, value in
                         zip(('zero', 'mid', 'limit'), (0, .25, .5))]}}
    info = SimpleNamespace(joints={'joint': {'limits': [0, 2]}})
    assert state_commands(case, 'candidate', info)[-1][1] == {'joint': .5}
    case['replay_command_policy'] = 'current_joint_limits'
    assert state_commands(case, 'candidate', info)[-1][1] == {'joint': 2}
    assert state_commands(case, 'candidate', info)[1][1] == {'joint': 1}
    info.joints = {}
    assert all(not values for _, values in state_commands(case, 'candidate', info))


@pytest.fixture
def fake_session(tmp_path, monkeypatch):
    import turnitover.render.session as session_module
    import turnitover.render.views as views_module
    monkeypatch.syspath_prepend(str(tmp_path))  # Undo run's native-root insertion afterwards.

    class Session:
        failure = None
        def __init__(self, *args):
            self._page = SimpleNamespace(evaluate=lambda *args: {'geometries': 0})
            self.info = SimpleNamespace(parts=('part',), joints={})
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def browser_version(self): return 'offline-fixture'
        def load(self, program, *args):
            self.source = program.source
            if self.failure and program.source == 'candidate':
                raise self.failure
        def reset_joints(self): pass
        def set_joints(self, values): pass
        def request_view(self, view): return SimpleNamespace(png=b'image')
        def dispose_program(self): return {'ok': True}
        def _call(self, method, *args):
            return {'parts': [mesh()]} if method == 'exportGeometry' else {'draw_calls': 1, 'triangles_scene': 4}

    monkeypatch.setattr(session_module, 'ObservationSession', Session)
    monkeypatch.setattr(views_module, 'load_views', lambda _: ())
    return Session


def bridge_input(tmp_path):
    ref = tmp_path/'reference.ts'; ref.write_text('reference', encoding='utf-8')
    cand = tmp_path/'candidate.ts'; cand.write_text('candidate', encoding='utf-8')
    case = {'reference_program': str(ref), 'view': 'front',
        'framing': {'center': [0, 0, 0], 'radius': 1},
        'commands': [[state, {}] for state in ('zero', 'mid', 'limit')]}
    return case, cand


def test_native_bridge_collects_private_geometry_without_adding_it_to_runtime(tmp_path, fake_session):
    case, candidate = bridge_input(tmp_path)
    result = run(tmp_path, case, candidate, tmp_path/'out')
    assert result['status'] == 'complete'
    assert set(result['private_evidence']) == {'reference', 'candidate'}
    assert 'positions_b64' not in json.dumps(result['runtime'])
    context = {'max_triangles': 100, 'max_draw_calls': 10}
    result = measure_replay(result, [{'defect_id': 'structure.part_offset', 'parts': ['part']}], context)
    assert result['measurements'][0]['passed'] is True
    assert SCORING_CONTRACT['status'] == 'development'


def test_native_bridge_classifies_typescript_error_as_candidate_gate(tmp_path, fake_session):
    from turnitover.render.transpile import CompileError
    fake_session.failure = CompileError(f'Expected semicolon {tmp_path}')
    case, candidate = bridge_input(tmp_path)
    result = run(tmp_path, case, candidate, tmp_path/'out')
    assert result['status'] == 'complete' and result['gate']['passed'] is False
    assert result['gate']['stage'] == 'compile'
    assert str(tmp_path) not in result['gate']['diagnostic']
    assert 'Expected semicolon' in result['gate']['diagnostic']


def test_native_bridge_does_not_turn_transport_failure_into_candidate_failure(tmp_path, fake_session):
    from turnitover.render.session import BrowserTransportError
    fake_session.failure = BrowserTransportError('Browser disconnected')
    case, candidate = bridge_input(tmp_path)
    with pytest.raises(BrowserTransportError):
        run(tmp_path, case, candidate, tmp_path/'out')
    assert json.loads((tmp_path/'out/replay.json').read_text())['status'] == 'incomplete'


def test_native_bridge_deterministic_candidate_runtime_rejection_is_gate(tmp_path, fake_session):
    from turnitover.render.session import HarnessError
    def bad_tick(self, view):
        if self.source == 'candidate':
            raise HarnessError('render', 'invalid candidate object')
        return SimpleNamespace(png=b'image')
    fake_session.request_view = bad_tick
    case, candidate = bridge_input(tmp_path)
    result = run(tmp_path, case, candidate, tmp_path/'out')
    assert result['status'] == 'complete' and result['gate']['passed'] is False
    assert result['gate']['stage'] == 'render'


def material_readback():
    texture = {'complete': True, 'rgba_sha256': 'a' * 64, 'width': 2, 'height': 2,
        'repeat': [1, 1], 'offset': [0, 0], 'center': [0, 0], 'rotation': 0,
        'matrix': [1, 0, 0, 0, 1, 0, 0, 0, 1], 'matrixAutoUpdate': True,
        'wrapS': 1000, 'wrapT': 1000, 'channel': 0}
    return {'type': 'MeshStandardMaterial',
        'properties': {**dict.fromkeys(PROPERTIES), 'roughness': .5, 'metalness': 0,
                       'normalScale': [1, 1], 'normalMapType': 0,
                       'emissive': [0, 0, 0], 'emissiveIntensity': 1},
        'textures': {**dict.fromkeys(TEXTURES), 'map': texture, 'normalMap': deepcopy(texture)},
        'gpu': {'program_present': True, 'roughness': .5, 'metalness': 0,
                'emissive': [0, 0, 0], 'normalScale': [1, 1],
                **dict.fromkeys(('uses_roughnessmap', 'uses_metalnessmap', 'uses_emissivemap'), False),
                'uses_normalmap': True},
        'roughness_and_basecolor_share_texture': False}


@pytest.fixture
def property_native(native):
    source, docs, replay = native
    hierarchy = {'property': 'hierarchy', 'root': {'name': 'root', 'children': [
        {'name': 'part', 'children': []}, {'name': 'other', 'children': []}]}}
    for doc in docs.values():
        doc['initial'] = {'tick_callback_present': False}
        for state in doc['states'].values():
            state['readback'] = {'schema_version': 1, 'hierarchy': deepcopy(hierarchy), 'meshes': [
                {'part': part, 'mesh': part + '#mesh', 'supported_geometry': True,
                 'attributes': {**dict.fromkeys(('uv1', 'uv2', 'normal', 'tangent')),
                                'uv': {'sha256': 'c' * 64, 'count': 4, 'itemSize': 2}},
                 'materials': [material_readback()]} for part in ('part', 'other')]}
        doc['lifecycle'] = {'resources': [{'uuid': 'geometry', 'kind': 'geometry'},
                                        {'uuid': 'material', 'kind': 'material'}],
            'events': [{'uuid': key, 'active_object_null': True, 'root_detached': True}
                       for key in ('geometry', 'material')],
            'active_object_null': True, 'root_detached': True,
            'result': {'ok': True, 'callback_present': True, 'callback_error': None},
            'before_load': {'geometries': 0, 'textures': 0},
            'before_dispose': {'geometries': 1, 'textures': 0},
            'after_dispose': {'geometries': 0, 'textures': 0}}
    return source, docs, replay


@pytest.mark.parametrize('kind,group,key,value', [
    ('material.basecolor_as_rough', 'textures', 'roughnessMap', {'complete': True, 'rgba_sha256': 'b' * 64}),
    ('material.metalness', 'properties', 'metalness', 1),
    ('material.emissive', 'properties', 'emissiveIntensity', 4),
    ('material.normal_map', 'properties', 'normalScale', [-1, -1]),
    ('material.tiling_scale', 'textures', 'map', {**material_readback()['textures']['map'], 'repeat': [4, 4]}),
])
def test_native_material_fields_detect_each_corruption(property_native, kind, group, key, value):
    source, docs, replay = property_native
    audit = make_audit(audit_case(kind), lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    docs['candidate']['states']['mid']['readback']['meshes'][0]['materials'][0][group][key] = value
    result = audit(source)
    assert result['defects'][0]['resolved'] is False
    assert result['program_checks']['material.properties'] is False
    assert json.loads(json.dumps(result))['passed'] is False  # Actual audit artifacts must serialize.


def test_native_material_checks_actual_gpu_with_cpu_values_still_correct(property_native):
    source, docs, replay = property_native
    docs['candidate']['states']['limit']['readback']['meshes'][0]['materials'][0]['gpu']['metalness'] = 1
    result = make_audit(audit_case('material.metalness'), lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is False


def test_material_resolution_does_not_hide_another_part_regression(property_native):
    source, docs, replay = property_native
    docs['candidate']['states']['mid']['readback']['meshes'][1]['materials'][0]['properties']['metalness'] = 1
    result = make_audit(audit_case('material.metalness'), lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is True
    assert result['program_checks']['material.properties'] is False
    assert result['passed'] is False


def test_unreadable_texture_cannot_pass_material_check(property_native):
    source, docs, replay = property_native
    docs['candidate']['states']['mid']['readback']['meshes'][0]['materials'][0]['textures']['map']['complete'] = False
    result = make_audit(audit_case('material.tiling_scale'), lambda _: replay())(source)
    assert result['program_checks']['material.properties'] is None


def test_absent_reference_normal_map_cannot_establish_normal_map_restoration(property_native):
    source, docs, replay = property_native
    for doc in docs.values():
        for state in doc['states'].values():
            material = state['readback']['meshes'][0]['materials'][0]
            material['textures']['normalMap'] = None
            material['gpu']['uses_normalmap'] = False
    result = make_audit(audit_case('material.normal_map'), lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is None


def test_private_direct_time_replay_cannot_hide_material_or_disposal_regression(property_native):
    source, docs, replay = property_native
    for doc in docs.values():
        doc['direct_time_endpoint'] = deepcopy(doc['states']['limit'])
        doc['direct_time_lifecycle'] = deepcopy(doc['lifecycle'])
    material = docs['candidate']['direct_time_endpoint']['readback']['meshes'][0]['materials'][0]
    material['properties']['metalness'] = 1
    docs['candidate']['direct_time_lifecycle']['events'] = []
    result = make_audit(audit_case('material.metalness'), lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is False
    assert result['program_checks']['runtime.lifecycle'] is False


def test_wrong_parent_detected_even_when_all_world_geometry_is_identical(property_native):
    source, docs, replay = property_native
    audit = make_audit(audit_case('structure.wrong_parent'), lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    root = docs['candidate']['states']['mid']['readback']['hierarchy']['root']
    root['children'][1]['children'].append(root['children'].pop(0))
    result = audit(source)
    assert result['program_checks']['geometry.alignment'] is True
    assert result['defects'][0]['resolved'] is False


@pytest.mark.parametrize('change', ['noop', 'leak', 'double', 'attached', 'missing'])
def test_owned_resource_disposal_requires_events_detachment_and_memory(property_native, change):
    source, docs, replay = property_native
    audit = make_audit(audit_case('runtime.missing_dispose'), lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    value = docs['candidate']['lifecycle']
    if change == 'noop': value['events'] = []
    elif change == 'leak': value['after_dispose']['geometries'] = 1
    elif change == 'double': value['events'].append(deepcopy(value['events'][0]))
    elif change == 'attached': value['events'][0]['root_detached'] = False
    else: del docs['candidate']['lifecycle']
    result = audit(source)
    assert result['defects'][0]['resolved'] is (None if change == 'missing' else False)


def test_original_authored_history_is_not_replaced_with_only_three_states(property_native):
    source, docs, replay = property_native
    for doc in docs.values():
        doc['authored_history'] = {'complete': True, 'snapshots': [deepcopy(doc['states']['zero'])],
                                   'lifecycle': deepcopy(doc['lifecycle'])}
    material = docs['candidate']['authored_history']['snapshots'][0]['readback']['meshes'][0]['materials'][0]
    material['properties']['metalness'] = 1
    result = make_audit(audit_case('material.metalness'), lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is False


def test_late_return_duplicate_is_measured_and_not_given_as_public_feedback(property_native):
    source, docs, replay = property_native
    for doc in docs.values():
        doc['return_to_zero'] = deepcopy(doc['states']['zero'])
    exported = docs['candidate']['return_to_zero']['geometry']['parts'][0]
    faces = base64.b64decode(exported['indices_b64'])
    exported['indices_b64'] = base64.b64encode(faces + faces).decode()
    exported['triangle_count'] *= 2
    result = make_audit(audit_case('geometry.duplicate_part'), lambda _: replay())(source)
    assert result['defects'][0]['resolved'] is False
    assert result['program_checks']['geometry.topology'] is False
    assert 'return_to_zero' not in json.dumps(replay()['runtime'])


def test_open_volume_detects_missing_face_while_reusing_actual_export(native):
    source, docs, replay = native
    audit = make_audit(audit_case('geometry.open_volume'), lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    exported = docs['candidate']['states']['mid']['geometry']['parts'][0]
    exported['indices_b64'] = base64.b64encode(base64.b64decode(exported['indices_b64'])[:-12]).decode()
    exported['triangle_count'] -= 1
    assert audit(source)['defects'][0]['resolved'] is False


def test_native_proper_crossing_and_budget_unknown():
    from turnitover.checkers.base import PartMesh
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0],
                         [0, -.5, -1], [0, -.5, 1], [0, .5, 0.]])
    value = PartMesh(vertices, np.array([[0, 1, 2], [3, 4, 5]]))
    result = self_crossings(value)
    assert result['complete'] and result['count'] == 1
    assert self_crossings(value, max_faces=1)['complete'] is False
    assert self_crossings(value, max_pairs=0)['complete'] is False


def test_absolute_time_history_checks_missing_callback_phase_and_accumulation(property_native):
    source, docs, replay = property_native
    for doc in docs.values():
        doc['initial']['tick_callback_present'] = True
        doc['time_sequence'] = [0., .5, 1.]
        doc['states']['mid']['geometry']['parts'][0] = mesh(offset=.3)
        doc['states']['limit']['geometry']['parts'][0] = mesh(offset=.5)
        doc['direct_time_endpoint'] = {'time_seconds': 1., 'geometry': deepcopy(doc['states']['limit']['geometry'])}
    audit = make_audit(audit_case('runtime.tick_missing_or_phase'), lambda _: replay())
    assert audit(source)['defects'][0]['resolved'] is True
    docs['candidate']['direct_time_endpoint']['geometry']['parts'][0] = mesh(offset=.1)
    assert audit(source)['defects'][0]['resolved'] is False
    docs['candidate']['direct_time_endpoint'] = deepcopy(docs['reference']['direct_time_endpoint'])
    docs['candidate']['states']['mid']['geometry']['parts'][0] = mesh(offset=.1)
    assert audit(source)['defects'][0]['resolved'] is False
    docs['candidate']['initial']['tick_callback_present'] = False
    assert audit(source)['defects'][0]['resolved'] is False


@pytest.fixture
def gold_observations(tmp_path):
    from turnitover.repair.table1_judges import ObservationPair, make_public_record
    pairs = [ObservationPair(state, tmp_path / f'{state}-ref.png', tmp_path / f'{state}-cand.png')
             for state in ('zero', 'mid', 'limit')]
    context = {'task': 'Match reference', 'parts': ['part', 'other'], 'joints': ['joint'],
               'views': ['front'], 'runtime_properties': ['stats'],
               'action_types': ['request_view', 'query_runtime'],
               'max_triangles': 100, 'max_draw_calls': 10}
    return {'pairs': pairs, 'public_record': make_public_record(pairs, [], context)}


def test_gold_rechecks_current_program_and_removes_repaired_findings(property_native, gold_observations):
    from turnitover.repair.table1_judges import evaluate_judge
    source, docs, replay = property_native
    case = audit_case()
    case['private_labels'][0]['params'] = {'private_marker': 'SECRET_PARAMETER'}
    gold = make_gold(case, lambda _: replay())
    docs['candidate']['states']['mid']['geometry']['parts'][0] = mesh(offset=.2)
    def verdict():
        return evaluate_judge('gold', gold_observations['pairs'],
            public_record=gold_observations['public_record'],
            gold_call=lambda: gold(source, gold_observations))
    broken = verdict()
    assert broken['status'] == 'fail'
    assert broken['feedback']['findings'][0]['evidence_steps'] == (1,)
    assert broken['feedback']['findings'][0]['defect_id'] == 'structure.part_offset'
    text = json.dumps(broken)
    assert all(private not in text for private in ('SECRET_PARAMETER', 'positions_b64', source,
                                                   'private_evidence', 'sha256', 'surface_error'))
    docs['candidate']['states']['mid']['geometry']['parts'][0] = mesh()
    repaired = verdict()
    assert repaired['status'] == 'pass'
    assert not repaired['feedback']['findings']


def test_gold_unattributed_regression_or_unknown_is_uncertain(property_native, gold_observations):
    source, docs, replay = property_native
    gold = make_gold(audit_case(), lambda _: replay())
    docs['candidate']['states']['mid']['readback']['meshes'][1]['materials'][0]['properties']['metalness'] = 1
    result = gold(source, gold_observations)
    assert result['status'] == 'uncertain' and not result['findings']
    docs['candidate']['states']['mid']['readback']['meshes'][1]['materials'][0]['properties']['metalness'] = 0
    del docs['candidate']['lifecycle']
    assert gold(source, gold_observations)['status'] == 'uncertain'


def test_gold_preserves_current_source_and_private_evidence_hash_checks(native, gold_observations):
    source, _, replay = native
    evidence = replay()
    gold = make_gold(audit_case(), lambda _: evidence)
    with pytest.raises(ValueError, match='current-source'):
        gold('stale source', gold_observations)
    from pathlib import Path
    Path(evidence['private_evidence']['candidate']['path']).write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        gold(source, gold_observations)


def test_attribute_profiles_allow_remesh_but_reject_changed_mapping():
    from turnitover.repair.native_properties import _attribute
    profile = {'mean': [.5, .5], 'second': [.333, .333],
               'position_cross': [.05, 0, 0, 0, .05, 0]}
    original = {'sha256': 'a' * 64, 'count': 4, 'itemSize': 2, 'profile': profile}
    remesh = {**deepcopy(original), 'sha256': 'b' * 64, 'count': 9}
    assert _attribute(original, remesh) is True
    remesh['profile']['position_cross'][0] *= -1
    assert _attribute(original, remesh) is False
    remesh['profile']['position_cross'][0] *= -1
    remesh['profile']['mean'][0] += 1
    assert _attribute(original, remesh) is False
    remesh['profile']['mean'][0] = float('nan')
    assert _attribute(original, remesh) is None
    remesh['profile']['mean'][0] = None  # Browser JSON representation of a nonfinite value.
    assert _attribute(original, remesh) is None
