import json

from scripts.score_motion_pixel_control import score


def frozen_case(kind='offset', condition='frozen_B/full_motion'):
    states = [{'step': i, 'view': 'front', 'actual_value': i} for i in range(3)]
    source = {'id': 'x', 'asset_id': 'object', 'condition': condition,
              'messages': [{'content': 'Public input: '+json.dumps({'states': states})}],
              'evidence_target': {'defect_type': kind, 'joint': None, 'part': 'wrong'},
              'gold': {'latent_class': kind, 'comparison': {'static_steps': []}}}
    gold = {('object', kind): {'class': kind, 'joint': 'joint_0' if kind == 'axis' else None,
                              'part': 'link_2' if kind != 'clean' else None,
                              'pixel_signals': [{'exposed': True}]*3}}
    return source, gold


def test_frozen_offset_uses_original_exposure_not_eroded_proxy():
    source, gold = frozen_case()
    prediction = {'raw': json.dumps({'defect_type': 'offset', 'joint': None,
                                     'part': 'link_2', 'evidence_steps': [0]})}
    assert score(source, prediction, gold)['full_correct']


def test_axis_evidence_with_extra_nonexistent_step_is_rejected():
    source, gold = frozen_case('axis')
    prediction = {'raw': json.dumps({'defect_type': 'axis', 'joint': 'joint_0',
                                     'part': 'link_2', 'evidence_steps': [0, 1, 99]})}
    result = score(source, prediction, gold)
    assert not result['valid'] and not result['full_correct']


def test_rest_only_axis_abstention_does_not_require_latent_location():
    source, gold = frozen_case('axis', 'frozen_C/rest_only')
    prediction = {'raw': json.dumps({'defect_type': 'uncertain', 'joint': None,
                                     'part': None, 'evidence_steps': []})}
    assert score(source, prediction, gold)['full_correct']
