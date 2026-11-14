"""Recognition uses the shared pair inference contract without training changes."""
import importlib.util
import json
from pathlib import Path

from turnitover.oracle.inference_prompt import tokenize_inference_prompt
from tests.unit.test_snapshot_scoring import inputs


def module():
    path = Path(__file__).resolve().parents[2] / 'scripts/evaluate_qwen_episode_sft.py'
    spec = importlib.util.spec_from_file_location('observation_recognition', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def paired_record():
    record = inputs()[0][0]
    payload = json.loads(record['messages'][0]['content'].rsplit('\n', 1)[-1])
    paths = [f'{state}-{role}.png' for state in ('zero', 'mid', 'limit')
             for role in ('reference', 'candidate')]
    payload['history'] = [dict(step=i, state=state,
        action={'type': 'request_view', 'view_id': 'front'},
        reference_image_ref=paths[2*i], image_ref=paths[2*i+1])
        for i, state in enumerate(('zero', 'mid', 'limit'))]
    payload['image_order'] = paths
    payload['context']['views'] = ['front']
    payload['observation_format'] = 'reference_candidate_pairs_v1'
    record['images'] = paths
    record['messages'][0]['content'] = 'Protocol\n' + json.dumps(payload)
    return record


def test_recognition_calls_three_pairs_and_sums_actual_usage():
    record, calls = paired_record(), []
    def call(row, images):
        calls.append((row, images))
        assert len(images) == 2
        assert row['messages'][1]['content'] == ''
        verdict = dict(status='uncertain', confidence=0., findings=[],
                       summary='Limited', limitations=['Limited'])
        return {'raw': json.dumps({'verdict': verdict}),
                'input_tokens': 10, 'output_tokens': 3, 'elapsed_ms': 2.}
    row = module().predict_record(record, list(record['images']), call)
    assert len(calls) == row['model_calls'] == 3
    assert row['input_tokens'] == 30
    assert row['output_tokens'] == 9
    assert row['total_tokens'] == 39
    assert row['predicted_status'] == 'uncertain'
    assert [x['observation_id'] for x in row['observations']] == ['zero', 'midpoint', 'limit']


def test_withheld_removes_model_input_before_inference():
    record, calls = paired_record(), []
    def call(row, images):
        calls.extend(images)
        verdict = dict(status='pass', confidence=1., findings=[],
                       summary='No visible defect', limitations=[])
        return {'raw': json.dumps({'verdict': verdict})}
    row = module().predict_record(record, list(record['images']), call, withheld=('midpoint',))
    assert row['model_calls'] == 2
    assert all('mid-' not in image for image in calls)
    assert row['predicted_status'] == 'uncertain'


def test_inference_prompt_never_tokenizes_assistant_target():
    record = {'images': ['ref', 'candidate'], 'messages': [
        {'role': 'user', 'content': 'Public input'},
        {'role': 'assistant', 'content': 'PRIVATE_TARGET'}]}
    class Processor:
        def apply_chat_template(self, prompt, **kwargs):
            assert 'PRIVATE_TARGET' not in str(prompt)
            assert kwargs['enable_thinking'] is False
            return 'tokens'
    _, tokens = tokenize_inference_prompt(Processor(), record, ['ref', 'candidate'], model_type='qwen3_5')
    assert tokens == 'tokens'
