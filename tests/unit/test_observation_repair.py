import json

from tests.unit.test_paired_observations import paired_record
from turnitover.models.client import ModelResult
from turnitover.repair.iterative import _verification_feedback
from turnitover.repair.loop import propose_repair
from turnitover.repair.observation_verifier import make_observation_verifier


def test_fresh_repair_adapter_uses_shared_aggregation_and_pair_feedback(tmp_path):
    record = paired_record()
    payload = json.loads(record['messages'][0]['content'].rsplit('\n', 1)[-1])
    payload['observation_format'] = 'reference_candidate_pairs_runtime_v1'
    payload['history'].append({'step': 3, 'action': {'type': 'query_runtime', 'property': 'stats'},
                               'image_ref': None, 'payload': {'draw_calls': 3}})
    record['messages'][0]['content'] = 'Protocol\n' + json.dumps(payload)
    images = []
    for name in record['images']:
        path = tmp_path / name
        path.write_bytes(b'fixture')
        images.append(path)
    programs, calls = [], []
    def acquire(program, *args, **kwargs):
        programs.append(program)
        return record, images
    target = json.loads(record['messages'][1]['content'])
    target['verdict']['findings'][0]['evidence_steps'] = [0]
    def model_call(single, pair):
        assert len(pair) == 2
        calls.append(single)
        return {'raw': json.dumps(target), 'usage': {'total_tokens': 7}}
    verify = make_observation_verifier(acquire, model_call)
    result = verify('current source', tmp_path / 'verifier', None, None, None)
    assert programs == ['current source']
    assert len(calls) == len(result['model_calls']) == 3
    assert sum(c['usage']['total_tokens'] for c in result['model_calls']) == 21
    assert len(result['verdict']['findings']) == 1
    assert result['verdict']['findings'][0]['evidence_steps'] == [0, 1, 2]
    assert result['findings_metadata'][0]['evidence_observations'] == ['zero', 'midpoint', 'limit']
    packet = _verification_feedback('compare', (), result, tmp_path / 'verifier')
    assert len(packet.images) == 6
    assert packet.public['findings_metadata'] == result['findings_metadata']
    def generator(prompt, supplied):
        assert len(supplied) == 6
        payload = json.loads(prompt.rsplit('\n', 1)[-1])
        for row in payload['feedback']['observations']:
            assert row['reference_image_ref'] in payload['image_order']
            assert row['image_ref'] in payload['image_order']
        return ModelResult('{"stop":true}', 'fixture', {}, 'stop', 0)
    propose_repair('source', generator, packet, tmp_path / 'generator')
