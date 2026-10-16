import copy
import json

from scripts.easy_teacher_audit import audit_case, scope_coverage
from scripts.easy_teacher_data import PROTOCOL
from scripts.summarize_kimi_easy_interactive import call_usage


def _manifest():
    return {'reference_protocol': {'states': [
        {'label': {'detent': name, 'view': 'oblique_fl'},
         'state': {'joint_0': value, 'joint_1': 0.0}}
        for name, value in [('rest', 0.0), ('mid', .5), ('limit', 1.0)]]}}


def _observation(value, **kwargs):
    return {'bound': True, 'reference_comparable': True,
            'state': {'view': 'oblique_fl', 'joints': {'joint_0': value, 'joint_1': 0.0}}, **kwargs}


def test_motion_pass_requires_all_declared_reference_states():
    result = scope_coverage({}, _manifest(), [_observation(0), _observation(1)])
    assert result == {'rest': True, 'mid': False, 'limit': True}


def test_different_camera_or_other_joint_state_cannot_support_motion_pass():
    wrong_camera = _observation(.5)
    wrong_camera['state']['view'] = 'front'
    wrong_joint = _observation(.5)
    wrong_joint['state']['joints']['joint_1'] = .2
    rows = [_observation(0), _observation(1), wrong_camera, wrong_joint,
            _observation(.5, bound=False)]
    assert not scope_coverage({}, _manifest(), rows)['mid']


def test_numeric_state_alias_counts_without_inventing_an_extra_observation():
    manifest = _manifest()
    manifest['reference_protocol']['states'][1]['state']['joint_0'] = 0.0
    assert all(scope_coverage({}, manifest, [_observation(0), _observation(1)]).values())


def test_scoped_pass_extension_cannot_remove_other_audit_failures(tmp_path, monkeypatch):
    manifest = _manifest()
    manifest['checker'] = {'code_correct': True}
    episode = {'episode_id': 'fixture', 'context': {'task': 'fixed-scope'},
               'reference_protocol': PROTOCOL, 'verdict': {'status': 'pass', 'findings': []},
               'status': 'complete', 'trajectory': [], 'model_calls': [],
               'data_identity': {'asset_id': 'fixture'}, 'token_usage': {}}
    (tmp_path / 'episode.json').write_text(json.dumps(episode))
    evidence = {'action_audit': [_observation(v) for v in (0, .5, 1)],
                'accepted': False, 'localization': [], 'evidence_sufficient': False,
                'diagnosis_outcome': 'pass_on_clean',
                'rejection_reasons': [
                    'pass_scope_not_supported:task_scope_not_limited_to_an_auditable_single_reference_rule',
                    'reference_pixels_differ_from_frozen_protocol']}
    monkeypatch.setattr('scripts.easy_teacher_audit.audit_episode', lambda *a, **kw: {
        'pipeline_ok': True, 'teacher_outputs_legal': True,
        'independent_evidence_audit': copy.deepcopy(evidence)})
    result = audit_case(tmp_path, {'context': episode['context']},
                        {'target': {'status': 'pass', 'findings': []}}, {}, manifest)
    assert not result['strict_correct']
    assert result['failure_reasons'] == ['reference_pixels_differ_from_frozen_protocol']


def test_motion_scope_cannot_approve_an_unobserved_mid_state(tmp_path, monkeypatch):
    manifest = _manifest()
    manifest['checker'] = {'code_correct': True}
    episode = {'episode_id': 'fixture', 'context': {'task': 'fixed-scope'},
               'reference_protocol': PROTOCOL, 'verdict': {'status': 'pass', 'findings': []},
               'status': 'complete', 'trajectory': [], 'model_calls': [],
               'data_identity': {'asset_id': 'fixture'}, 'token_usage': {}}
    (tmp_path / 'episode.json').write_text(json.dumps(episode))
    monkeypatch.setattr('scripts.easy_teacher_audit.audit_episode', lambda *a, **kw: {
        'pipeline_ok': True, 'teacher_outputs_legal': True, 'independent_evidence_audit': {
            'action_audit': [_observation(0), _observation(1)], 'accepted': False,
            'localization': [], 'evidence_sufficient': False, 'diagnosis_outcome': 'pass_on_clean',
            'rejection_reasons': ['pass_scope_not_supported:task_scope_not_limited_to_an_auditable_single_reference_rule']}})
    result = audit_case(tmp_path, {'context': episode['context']},
                        {'target': {'status': 'pass', 'findings': []}}, {}, manifest)
    assert not result['strict_correct'] and not result['scope_coverage']['mid']


def test_no_text_usage_is_recovered_only_from_recorded_provider_error():
    usage = {'prompt_tokens': 2918, 'completion_tokens': 4096, 'total_tokens': 7014}
    assert call_usage({'error_message': 'no text; finish_reason=length; usage=' + json.dumps(usage)}) == (
        usage, 'original_provider_error_message')
    assert call_usage({'error_message': 'connection failed'}) == ({}, 'unavailable')
