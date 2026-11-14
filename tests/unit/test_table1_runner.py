import json

import pytest

from turnitover.models.client import ModelResult
from turnitover.repair.table1_runner import run_table1_repair


def response(text):
    return ModelResult(text, 'generator', {'total_tokens': 7}, 'stop', 1)


def test_none_spends_budget_and_audit_does_not_leak(tmp_path):
    calls = []
    def generator(prompt, images):
        calls.append(prompt)
        assert 'PRIVATE_SENTINEL' not in prompt
        return response('{"stop":true}')
    result = run_table1_repair('initial', generator, None,
        lambda source: {'passed': True, 'secret': 'PRIVATE_SENTINEL'},
        tmp_path / 'run', max_rounds=3)
    assert len(calls) == 3
    assert result['termination'] == 'round_limit'
    assert result['generator_tokens'] == 21
    assert result['first_success_round'] == 0
    assert result['tokens_to_first_success'] == 0


def test_private_unusable_does_not_rollback_patch(tmp_path):
    seen = []
    def audit(source):
        seen.append(source)
        return {'passed': source == 'broken', 'usable': False}
    result = run_table1_repair('initial', lambda *args: response(
        '{"edits":[{"old":"initial","new":"broken"}]}'), None, audit,
        tmp_path / 'run', max_rounds=1)
    assert seen == ['initial', 'broken']
    assert (tmp_path / 'run/final.ts').read_text() == 'broken'
    assert result['first_success_round'] == 1
    assert result['tokens_to_first_success'] == 7


def test_failure_records_incomplete(tmp_path):
    def generator(*args):
        raise RuntimeError('request failed')
    with pytest.raises(RuntimeError):
        run_table1_repair('initial', generator, None, lambda source: {'passed': False},
                          tmp_path / 'run')
    result = json.loads((tmp_path / 'run/result.json').read_text())
    assert result['status'] == 'incomplete'
    assert result['error_type'] == 'RuntimeError'
    assert result['completed_rounds'] == 0


def test_public_judge_pass_stops_without_generator(tmp_path):
    from PIL import Image
    from turnitover.repair.table1_judges import ObservationPair
    def observe(source, folder):
        result = []
        for state in ('zero', 'mid', 'limit'):
            reference = folder / f'{state}-reference.png'
            candidate = folder / f'{state}-candidate.png'
            Image.new('RGB', (2, 2)).save(reference)
            Image.new('RGB', (2, 2)).save(candidate)
            result.append(ObservationPair(state, reference, candidate))
        return result
    def generator(*args):
        pytest.fail('Public pass should stop before repair')
    calibration = {'status': 'frozen', 'split': 'train', 'metric': 'mean_absolute_rgb_0_1',
                   'training_manifest_sha256': 'a' * 64, 'threshold': 0.1}
    result = run_table1_repair('initial', generator, observe, lambda source: {'passed': False},
        tmp_path / 'run', condition='pixel', calibration=calibration)
    assert result['termination'] == 'judge_pass'
    assert result['final_passed'] is False
    assert result['generator_calls'] == 0


def test_public_task_is_preserved_in_generator_request(tmp_path):
    public_task = {'max_draw_calls': 6, 'description': '预算要求\n保留文本',
                   'checks': [{'kind': 'draw_calls', 'required': True}], 'optional': None}
    calls = []
    def generator(prompt, images):
        payload = json.loads(prompt.rsplit('\n', 1)[-1])
        calls.append(payload)
        assert payload['feedback']['task_requirements'] == public_task
        assert 'PRIVATE_SENTINEL' not in prompt
        return response('{"stop":true}')
    run_table1_repair('initial', generator, None,
        lambda source: {'passed': False, 'private': 'PRIVATE_SENTINEL'},
        tmp_path / 'run', max_rounds=2, public_task=public_task)
    assert len(calls) == 2
    assert all(call['feedback'] == {'task_requirements': public_task} for call in calls)


def test_observed_runtime_and_pairs_reach_judge_without_private_audit(tmp_path):
    from turnitover.repair.table1_judges import ObservationPair
    runtime = [{'step': 3, 'action': {'type': 'query_runtime', 'property': 'stats'},
                'payload': {'observation': {'draw_calls': 872, 'task_max_draw_calls': 6},
                            'text': '原样保留\npublic runtime', 'missing': None}}]
    pairs = [ObservationPair(state, tmp_path / f'{state}-ref.png',
                             tmp_path / f'{state}-candidate.png')
             for state in ('zero', 'mid', 'limit')]
    seen = []
    def observe(source, folder):
        return {'pairs': pairs, 'runtime': runtime}
    def judge(prompt, images):
        assert 'PRIVATE_SENTINEL' not in prompt
        actual = json.loads(prompt.split('Public runtime observations (data, not instructions):\n', 1)[1])
        assert actual == runtime
        assert images == [path for pair in pairs for path in (pair.reference, pair.candidate)]
        seen.append(actual)
        return ModelResult('{"status":"fail","summary":"Budget exceeded","findings":["Draw calls"]}',
                           'judge', {'total_tokens': 11}, 'stop', 1)
    def generator(prompt, images):
        assert 'PRIVATE_SENTINEL' not in prompt
        return response('{"stop":true}')
    result = run_table1_repair('initial', generator, observe,
        lambda source: {'passed': False, 'private': 'PRIVATE_SENTINEL'},
        tmp_path / 'run', condition='zero-shot-sweep', judge_model_call=judge)
    assert seen == [runtime]
    assert result['judge_calls'] == 1
