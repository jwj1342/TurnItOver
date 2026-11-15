import json
from pathlib import Path

import pytest

from scripts import evaluate_oracle_qwen as evaluation
from turnitover.models.client import ModelResult


@pytest.mark.parametrize("predicted", [None, "kinematics.joint_axis", "structure.part_offset"])
def test_detection_metrics_and_gold_stays_out_of_prompt(tmp_path, monkeypatch, predicted):
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
            response = json.dumps({'verdict': dict(status='fail', confidence=1., summary='Visible defect', limitations=[],
                findings=[dict(defect_id=predicted, parts=['door'], severity=.5, confidence=1., evidence_steps=[0],
                               description='Visible deviation', suggested_fix='Restore reference geometry')])})
            return ModelResult(response if predicted else 'invalid', 'fake', {}, 'completed', 0.)
    monkeypatch.setattr(evaluation, 'LocalQwen', FakeModel)
    report = evaluation.run(oracle, paired, Path('fake'), tmp_path / 'results')
    for metric in report['metrics'].values():
        assert metric['invalid'] == int(predicted is None)
        matched = predicted == 'kinematics.joint_axis'
        assert metric['classes']['kinematics.joint_axis'] == dict(
            tp=int(matched), fp=0, fn=int(not matched), precision=1. if matched else None,
            recall=float(matched), f1=float(matched))
        if predicted == 'structure.part_offset':
            assert metric['classes'][predicted] == dict(tp=0, fp=1, fn=0, precision=0., recall=None, f1=0.)


def test_readouts_keep_image_conditions_and_invalid_misses(tmp_path, monkeypatch):
    from scripts import evaluate_defect_readout, evaluate_pair_readout

    oracle, paired = tmp_path / 'oracle', tmp_path / 'paired'
    for root in (oracle, paired):
        root.mkdir()
        (root / 'manifest.json').write_text('{"status":"complete"}')
    label = 'kinematics.joint_axis'
    annotation = dict(id='case', split='test', labels=[dict(defect_id=label)], grid=[{}, {}, {}],
                      exposures=[[], [], [0]], trajectories={'oracle': {'witness_indices': [0, 2]}},
                      signals={'against_reference': [dict(image_l1=0, exposed=False),
                                                     dict(image_l1=1, exposed=True),
                                                     dict(image_l1=.5, exposed=True)]})
    (oracle / 'annotations.private.jsonl').write_text(json.dumps(annotation))
    (paired / 'test.diagnosis.jsonl').write_text(json.dumps(dict(
        id='case:oracle:paired', images=['front-ref.png', 'front.png', 'best-ref.png', 'best.png'])))
    calls = []
    responses = iter(['invalid', json.dumps({'defect_ids': [label]}),
                      '{"different":false}', '{"different":true}', 'invalid'])

    def model(prompt, paths):
        calls.append([p.name for p in paths])
        return ModelResult(next(responses), 'fake', {}, 'completed', 0.)

    for module in (evaluate_defect_readout, evaluate_pair_readout):
        monkeypatch.setattr(module, 'LocalQwen', lambda *args, **kwargs: model)
    defects = evaluate_defect_readout.run(oracle, paired, Path('fake'), tmp_path / 'defects')['metrics']
    assert defects['single']['invalid'] == 1 and defects['single']['classes'][label]['fn'] == 1
    assert defects['paired']['classes'][label]['tp'] == 1
    pairs = evaluate_pair_readout.run(oracle, paired, Path('fake'), tmp_path / 'pairs')['metrics']
    assert pairs['front']['correct'] == pairs['strongest']['tp'] == pairs['identical']['invalid'] == 1
    assert calls == [['front-ref.png', 'front.png', 'best.png'],
                     ['front-ref.png', 'front.png', 'best-ref.png', 'best.png'],
                     ['front-ref.png', 'front.png'], ['best-ref.png', 'best.png'], ['best.png', 'best.png']]
