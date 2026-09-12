import json
from pathlib import Path

from scripts import evaluate_oracle_qwen as evaluation
from turnitover.models.client import ModelResult


def test_invalid_outputs_count_as_misses_and_gold_stays_out_of_prompt(tmp_path, monkeypatch):
    oracle, paired, benchmark = [tmp_path / name for name in ('oracle', 'paired', 'benchmark')]
    oracle.mkdir(); paired.mkdir(); (benchmark / 'prepared').mkdir(parents=True)
    (oracle / 'manifest.json').write_text(json.dumps(dict(status='complete', benchmark=str(benchmark))))
    (paired / 'manifest.json').write_text(json.dumps(dict(status='complete')))
    context = dict(task='verify', parts=['door'], joints=['hinge'], views=['front'],
                   runtime_properties=[], action_types=['request_view'], max_triangles=5000, max_draw_calls=32)
    (benchmark / 'prepared/test.inputs.jsonl').write_text(json.dumps(dict(id='case', context=context)))
    (benchmark / 'prepared/test.gold.jsonl').write_text(json.dumps(dict(id='case', target=dict(findings=[dict(defect_id='kinematics.joint_axis', secret='PRIVATE_GOLD')]))))
    (paired / 'test.diagnosis.jsonl').write_text(json.dumps(dict(id='case:oracle:paired', images=[], messages=[dict(content='paired prompt')])) )
    (oracle / 'annotations.private.jsonl').write_text(json.dumps(dict(id='case', split='test', trajectories=dict(oracle=dict(history=[dict(step=0, action=dict(type='request_view', view_id='front'), image_ref='c.png')])))))
    class FakeModel:
        def __init__(self, path): pass
        def __call__(self, prompt, paths):
            assert 'PRIVATE_GOLD' not in prompt
            return ModelResult('invalid', 'fake', {}, 'completed', 0.)
    monkeypatch.setattr(evaluation, 'LocalQwen', FakeModel)
    report = evaluation.run(oracle, paired, Path('fake'), tmp_path / 'results')
    for metric in report['metrics'].values():
        assert metric['invalid'] == 1
        assert metric['classes']['kinematics.joint_axis']['fn'] == 1
        assert metric['classes']['kinematics.joint_axis']['recall'] == 0.
