import hashlib
import json

import pytest
from PIL import Image

from turnitover.models.client import ModelResult
from turnitover.repair.table1_judges import (
    CONDITIONS, ObservationPair, evaluate_judge, feedback_images, make_public_record,
)
from turnitover.repair.table1_scoring import (
    checker_coverage, classify_exposure, make_repair_audit, score_audit,
    summarize_condition, summarize_table1,
    runtime_budget_predicate,
)


@pytest.fixture
def observations(tmp_path):
    pairs = []
    for state in ('zero', 'mid', 'limit'):
        paths = [tmp_path / f'{state}-{role}.png' for role in ('reference', 'candidate')]
        for path in paths:
            Image.new('RGB', (2, 2), 'black').save(path)
        pairs.append(ObservationPair(state, *paths))
    context = {'task': 'Match reference', 'parts': ['part'], 'joints': ['joint'],
               'views': ['front'], 'runtime_properties': ['stats'],
               'action_types': ['request_view', 'query_runtime'],
               'max_triangles': 100, 'max_draw_calls': 10}
    runtime = [{'state': 'mid', 'kind': 'public_stats', 'value': {'draw_calls': 11}}]
    record = make_public_record(pairs, runtime, context)
    return pairs, runtime, record


def verdict(status='fail', step=1):
    findings = [] if status != 'fail' else [{'defect_id': 'structure.part_offset',
        'parts': ['part'], 'joint_id': None, 'severity': .5, 'confidence': .7,
        'evidence_steps': [step], 'description': 'Offset', 'suggested_fix': 'Align part'}]
    return {'status': status, 'confidence': .7, 'findings': findings,
            'summary': 'Compared', 'limitations': ['Insufficient'] if status == 'uncertain' else []}


@pytest.mark.parametrize('condition', ['zero-shot-single', 'zero-shot-sweep',
                                       'zero-shot-claude-sweep', 'ordinary', 'evidence'])
def test_all_model_conditions_use_fresh_paired_schema(condition, observations):
    pairs, runtime, record = observations
    seen = []
    def call(prompt, images):
        payload = json.loads(prompt.rsplit('\n', 1)[-1])
        measurements = [row['payload'] for row in payload['history'] if row['action']['type'] == 'query_runtime']
        assert prompt.startswith('Compare the supplied observations.')
        if condition == 'zero-shot-single' or payload['history'][0]['state'] != 'mid':
            assert measurements == []
        else:
            assert measurements == runtime
        assert 'assistant' not in payload
        seen.append(images)
        return ModelResult(json.dumps({'verdict': verdict(step=0)}), 'frozen',
                           {'total_tokens': 5}, 'stop', 1)
    result = evaluate_judge(condition, pairs, public_record=record, model_call=call)
    assert result['status'] == 'fail'
    assert len(seen) == (1 if condition == 'zero-shot-single' else 3)
    assert result['usage']['total_tokens'] == 5 * len(seen)
    if condition != 'zero-shot-single':
        assert all(len(images) == 2 for images in seen)
        assert result['feedback']['findings'][0]['evidence_steps'] == [0, 1, 2]
    else:
        assert len(seen[0]) == 2


def test_invalid_decisions_retry_then_uncertain_and_count_all_usage(observations):
    pairs, _, record = observations
    calls = []
    def invalid(*args):
        calls.append(args)
        return ModelResult('{"verdict":{}}', 'frozen', {'total_tokens': 7}, 'stop', 1)
    result = evaluate_judge('zero-shot-sweep', pairs, public_record=record, model_call=invalid)
    assert len(calls) == 12
    assert result['status'] == 'uncertain'
    assert result['usage']['total_tokens'] == 84
    assert not any(row['valid'] for row in result['attempts'])


def test_sweep_conditions_share_identical_model_inputs_and_aggregation(observations):
    pairs, _, record = observations
    observed, results = [], []
    for condition in ('zero-shot-sweep', 'zero-shot-claude-sweep', 'ordinary', 'evidence'):
        calls = []
        def call(prompt, images):
            calls.append((prompt, tuple(images)))
            return ModelResult(json.dumps({'verdict': verdict(step=0)}), 'same-model',
                               {'total_tokens': 5}, 'stop', 1)
        results.append(evaluate_judge(condition, pairs, public_record=record, model_call=call))
        observed.append(calls)
    assert len(observed[0]) == 3
    assert all(calls == observed[0] for calls in observed)
    assert all(result == results[0] for result in results)


def test_fresh_pair_view_comes_from_replay_recipe(observations):
    from turnitover.oracle.supervision_validation import public_payload
    pairs, runtime, record = observations
    context = {**public_payload(record)['context'], 'views': ['front', 'back']}
    with pytest.raises(ValueError, match='explicit view'):
        make_public_record(pairs, runtime, context)
    current = make_public_record(pairs, runtime, context, view_id='back')
    assert all(row['action']['view_id'] == 'back'
               for row in public_payload(current)['history'][:3])


def test_citations_deliver_actual_reference_candidate_pair(observations):
    pairs, _, record = observations
    def response(prompt, images):
        payload = json.loads(prompt.rsplit('\n', 1)[-1])
        status = 'fail' if payload['history'][0]['state'] == 'mid' else 'pass'
        return ModelResult(json.dumps({'verdict': verdict(status, step=0)}), 'model', {}, 'stop', 1)
    result = evaluate_judge('zero-shot-sweep', pairs, public_record=record, model_call=response)
    assert feedback_images(result, pairs) == (pairs[1].reference, pairs[1].candidate)
    with pytest.raises(ValueError, match='Gold requires'):
        evaluate_judge('gold', pairs, public_record=record)
    gold = evaluate_judge('gold', pairs, public_record=record, gold_call=lambda: verdict('pass'))
    assert gold['status'] == 'pass'
    assert len(CONDITIONS) == 10


@pytest.mark.parametrize('condition', ['ordinary-active', 'evidence-active'])
def test_active_conditions_select_x_of_three_with_zero_visible(condition, observations):
    pairs, _, record = observations
    calls = []
    def response(prompt, images):
        calls.append((prompt, tuple(images)))
        if prompt.startswith('Select additional joint-state observations'):
            assert tuple(images) == (pairs[0].reference, pairs[0].candidate)
            return ModelResult('{"observe":["mid"]}', 'selector',
                               {'total_tokens': 2}, 'stop', 1)
        payload = json.loads(prompt.rsplit('\n', 1)[-1])
        state = payload['history'][0]['state']
        return ModelResult(json.dumps({'verdict': verdict('fail', step=0)}), state,
                           {'total_tokens': 5}, 'stop', 1)
    result = evaluate_judge(condition, pairs, public_record=record, model_call=response)
    assert result['selection']['valid'] is True
    assert result['selection']['selected_states'] == ['zero', 'mid']
    assert len(calls) == 3  # one selection plus zero/mid diagnosis
    assert result['usage']['total_tokens'] == 12
    assert result['feedback']['findings'][0]['evidence_steps'] == [0, 1]
    assert {row['kind'] for row in result['attempts'] if 'kind' in row} == {'selection'}


def test_invalid_active_selection_falls_back_to_full_sweep(observations):
    pairs, _, record = observations
    calls = []
    def response(prompt, images):
        calls.append(prompt)
        if prompt.startswith('Select additional joint-state observations'):
            return ModelResult('{"verdict":{}}', 'selector',
                               {'total_tokens': 3}, 'stop', 1)
        return ModelResult(json.dumps({'verdict': verdict('pass')}), 'judge',
                           {'total_tokens': 5}, 'stop', 1)
    result = evaluate_judge('evidence-active', pairs, public_record=record,
                            model_call=response)
    assert result['selection']['valid'] is False
    assert result['selection']['fallback'] == 'all_states'
    assert result['selection']['selected_states'] == ['zero', 'mid', 'limit']
    assert len(calls) == 4
    assert result['usage']['total_tokens'] == 18
    assert result['status'] == 'pass'


def test_multi_defect_scoring_requires_each_instance_and_regression_checks():
    labels = [{'defect_id': 'structure.part_offset'}, {'defect_id': 'structure.part_offset'}]
    measurements = {0: {'passed': True, 'checker_id': 'native-offset-v1'}}
    result = score_audit(labels, measurements, {'abi': True}, ['abi'], program_sha256='current')
    assert result['passed'] is None
    assert result['missing_defects'] == [1]
    result = score_audit(labels, {**measurements, 1: measurements[0]}, {'abi': False}, ['abi'], program_sha256='current')
    assert result['passed'] is False
    assert all(row['resolved'] for row in result['defects'])
    assert not checker_coverage(['structure.part_offset'])['complete']
    assert len(checker_coverage([])['missing']) == 29


def test_audit_rejects_stale_replay_and_joins_callbacks():
    labels = [{'defect_id': 'structure.part_offset'}]
    source = 'candidate'
    replay = {'status': 'complete', 'program_sha256': {'candidate': hashlib.sha256(source.encode()).hexdigest()}}
    audit = make_repair_audit(lambda _: replay, labels,
        {'structure.part_offset': lambda *args: {'passed': True, 'checker_id': 'offset'}},
        {'abi': lambda _: True}, ['abi'])
    assert audit(source)['passed'] is True
    assert audit(source)['table1_eligible'] is False
    with pytest.raises(ValueError, match='current-source'):
        audit('edited')


def test_exposure_keeps_unknown_probe_and_temporal_separate():
    history = [{'step': 0, 'state': 'zero', 'action': {'type': 'request_view', 'view_id': 'front'}},
               {'step': 1, 'state': 'mid', 'action': {'type': 'request_view', 'view_id': 'front'}},
               {'step': 2, 'action': {'type': 'query_runtime'}},
               {'step': 3, 'action': {'type': 'query_runtime'}, 'payload': {'kind': 'public_tick_trace'}}]
    def classify(steps):
        return classify_exposure(0, {'frozen': True, 'label_evidence_steps': {'0': steps}}, history)
    assert classify([0, 1]) == 'visible_at_rest'
    assert classify([1]) == 'needs_motion'
    assert classify([2]) == 'probe'
    assert classify([3]) == 'temporal'
    assert classify([]) == classify([99]) == 'unknown'
    assert classify_exposure(0, {'label_evidence_steps': {'0': [0]}}, history) == 'unknown'
    history[1]['action']['view_id'] = 'back'
    assert classify([1]) == 'unknown'


def test_unknowns_preserve_denominators_and_cost_uses_final_successes_only():
    rows = [{'status': 'complete', 'final_passed': True, 'first_success_round': 1,
             'tokens_to_first_success': 12, 'total_tokens': 90},
            {'status': 'complete', 'final_passed': False, 'first_success_round': 1,
             'tokens_to_first_success': 8, 'total_tokens': 100}, {'status': 'incomplete'}]
    result = summarize_condition(rows)
    assert result['target'] == 3
    assert result['success_rate'] is None
    assert result['success_lower_bound'] == 1 / 3
    assert result['cost_to_success'] == {'runs': 1, 'iterations': 1, 'tokens': 12}
    paired = [{**row, 'id': str(i), 'condition': 'evidence', 'generator': g, 'repetition': 'r1',
               'comparison_protocol_sha256': 'a'*64, 'generator_config': {'model': g}}
              for g in ('kimi', 'claude') for i, row in enumerate(rows)]
    assert summarize_table1(paired, ['kimi', 'claude'], expected_conditions=['evidence'])['evidence']['cost_to_success']['tokens'] == 12
    with pytest.raises(ValueError, match='identical'):
        summarize_table1(paired[:-1], ['kimi', 'claude'], expected_conditions=['evidence'])
    paired[-1]['generator_config'] = {'model': 'changed'}
    with pytest.raises(ValueError, match='generator.*config'):
        summarize_table1(paired, ['kimi', 'claude'], expected_conditions=['evidence'])
    paired[-1]['generator_config'] = {'model': 'claude'}
    paired[-1]['comparison_protocol_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='protocol'):
        summarize_table1(paired, ['kimi', 'claude'], expected_conditions=['evidence'])


def test_runtime_checker_missing_stats_cannot_pass():
    from turnitover.checkers.base import Evidence
    from turnitover.checkers.runtime_budget import RuntimeBudgetChecker
    checker = RuntimeBudgetChecker(100, 10)
    for stats in ({}, {'triangles_scene': 10}, {'triangles_scene': 10, 'draw_calls': float('nan')}):
        result = checker.check(Evidence('sha', True, None, {}, stats, {}))
        assert result.passed is False and result.score is None


def test_runtime_adapter_checks_each_state_and_each_budget_separately():
    evidence = {'program_sha256': {'candidate': 'current'}, 'runtime': [
        {'role': 'candidate', 'state': state, 'kind': 'public_stats',
         'value': {'triangles_scene': 50, 'draw_calls': 11 if state == 'mid' else 5}}
        for state in ('zero', 'mid', 'limit')]}
    def check(kind):
        return runtime_budget_predicate({'defect_id': 'runtime.' + kind}, evidence,
                                        max_triangles=100, max_draw_calls=10)['passed']
    assert check('triangle_budget') is True
    assert check('draw_calls') is False
    evidence['runtime'].pop()
    assert check('triangle_budget') is None


@pytest.mark.parametrize('provider', ['anthropic', 'openai_compatible', 'openai', 'gemini'])
def test_judge_greedy_decoding_is_sent_to_provider(provider):
    from turnitover.models.client import request_payload
    from turnitover.models.config import ModelConfig
    config = ModelConfig('judge', provider, 'frozen', 'https://endpoint.test/v1', 'key', max_tokens=512)
    _, _, payload = request_payload(config, 'judge', [], temperature=0)
    options = payload['generationConfig'] if provider == 'gemini' else payload
    assert options['temperature'] == 0
