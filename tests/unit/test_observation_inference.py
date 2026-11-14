import json

import pytest

from turnitover.oracle.observation_inference import build_observation_inputs, infer_observationwise
from turnitover.oracle.paired_observations import user_content
from tests.unit.test_observation_recognition import paired_record


def record():
    row = paired_record()
    payload = json.loads(row['messages'][0]['content'].rsplit('\n', 1)[-1])
    payload['observation_format'] = 'reference_candidate_pairs_runtime_v1'
    payload['history'] += [dict(step=3+i, action={'type':'query_runtime','property':'stats'}, image_ref=None, saved_state=state, payload={'text':state or 'global'}) for i,state in enumerate(('mid',None))]
    row['messages'][0]['content'] = 'Protocol\n'+json.dumps(payload)
    return row


def answer(status='pass'):
    return json.dumps({'verdict':dict(status=status, confidence=.9, findings=[], summary='Observed', limitations=['Limited'] if status=='uncertain' else [])})


def test_single_pair_runtime_formatter_and_withheld():
    row=record()
    items=build_observation_inputs(row,row['images'])
    assert [x.observation_id for x in items]==['zero','midpoint','limit']
    for item in items:
        payload=json.loads(item.record['messages'][0]['content'].rsplit('\n',1)[-1])
        assert len(payload['history'])==(3 if item.observation_id=='midpoint' else 2)
        assert sum(x['type']=='image' for x in user_content(item.record,item.images))==2
    left=build_observation_inputs(row,row['images'],withheld=['midpoint'])
    assert len(left)==2
    assert all('mid-reference' not in str(x.record) and '"saved_state": "mid"' not in str(x.record) for x in left)


@pytest.mark.parametrize('bad',[None,{},'invalid'])
def test_invalid_missing_output_uncertain(bad):
    row=record();result=infer_observationwise(row,row['images'],lambda *args:bad)
    assert result['verdict']['status']=='uncertain'
    assert len(result['observations'])==3
    assert all(not x['valid'] for x in result['observations'])


def test_withheld_all_and_api_error_are_uncertain():
    row=record()
    result=infer_observationwise(row,row['images'],lambda *args:pytest.fail('No call'),withheld=['zero','midpoint','limit'])
    assert result['verdict']['status']=='uncertain' and result['observations']==[]
    def failed(*args):
        raise RuntimeError('Unavailable')
    result=infer_observationwise(row,row['images'],failed)
    assert len(result['observations'])==3 and result['verdict']['status']=='uncertain'
    assert result['observations'][0]['error_type']=='RuntimeError'


def test_pass_requires_all_three_observations():
    row=record()
    assert infer_observationwise(row,row['images'],lambda *args:answer())['verdict']['status']=='pass'
    assert infer_observationwise(row,row['images'],lambda *args:answer(),withheld=['limit'])['verdict']['status']=='uncertain'


def test_duplicate_findings_merge_real_sources_and_conflicting_suggestions():
    row=record(); count=[]
    payload=json.loads(row['messages'][0]['content'].rsplit('\n',1)[-1])
    context=payload['context']; part=context['parts'][0]
    def call(item, images):
        count.append(1)
        finding=dict(defect_id='structure.part_offset',parts=[part],joint_id=None,severity=.5,confidence=.8,evidence_steps=[0],description='Displaced',suggested_fix='Left' if len(count)==1 else 'Right')
        return {'raw':json.dumps({'verdict':dict(status='fail',confidence=.8,findings=[finding],summary='Mismatch',limitations=[])}),'usage':{'total_tokens':7}}
    result=infer_observationwise(row,row['images'],call)
    assert len(count)==3 and result['usage']['total_tokens']==21
    assert len(result['verdict']['findings'])==1
    assert result['verdict']['findings'][0]['evidence_steps']==[0,1,2]
    metadata=result['findings_metadata'][0]
    assert metadata['evidence_observations']==['zero','midpoint','limit']
    assert metadata['conflicts']==['Left','Right']


def test_cross_observation_citation_rejected():
    row=record()
    payload=json.loads(row['messages'][0]['content'].rsplit('\n',1)[-1])
    part=payload['context']['parts'][0]
    finding=dict(defect_id='structure.part_offset',parts=[part],joint_id=None,severity=.5,confidence=.8,evidence_steps=[99],description='Displaced',suggested_fix='Restore')
    raw=json.dumps({'verdict':dict(status='fail',confidence=.8,findings=[finding],summary='Mismatch',limitations=[])})
    result=infer_observationwise(row,row['images'],lambda *args:raw)
    assert result['verdict']['status']=='uncertain'
    assert all(not x['valid'] for x in result['observations'])
