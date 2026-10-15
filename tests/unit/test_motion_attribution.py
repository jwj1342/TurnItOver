"""Checks for attribution failure modes: rigid motion, missing evidence and false positives."""
import json
import numpy as np
from scripts.audit_motion_attribution import rigid_fit, rotation_distance
from scripts.score_motion_attribution import score


def gold(kind='axis'):
    return {'class':kind,'joint':'joint_1' if kind=='axis' else None,
            'part':'link_1' if kind!='clean' else None,
            'pixel_signals':[{'exposed':i not in (0,3) and kind!='clean'} for i in range(6)]}


def prediction(kind='axis',condition='B',steps=None):
    return {'id':'case','asset_id':'object','condition':condition,'truncated':False,
            'raw':json.dumps({'abnormal':kind=='axis','classification':kind,
                'joint_id':'joint_1' if kind=='axis' else None,
                'part_id':'link_1' if kind=='axis' else None,
                'evidence_steps':[0,1] if steps is None else steps,'explanation':'comparison'})}


def test_geometry_separates_rotation_from_translation():
    a=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
    expected=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    r,t,error=rigid_fit(a,a@expected.T+[.2,.3,.4])
    assert error<1e-10 and rotation_distance(r,expected)<1e-5
    r,_,_=rigid_fit(a,a+[.1,.2,.3])
    assert rotation_distance(r,np.eye(3))<1e-5


def test_lucky_axis_requires_visible_temporal_evidence():
    assert score(prediction(),gold(),{})['full_correct']
    r=score(prediction(steps=[0,3]),gold(),{})
    assert r['class_correct'] and not r['evidence_supports_type'] and not r['full_correct']


def test_clean_false_axis_is_unsupported_despite_valid_citations():
    r=score(prediction(),gold('clean'),{})
    assert r['citations_real'] and r['unsupported_finding'] and not r['class_correct']


def test_rest_only_clean_is_not_evidence_of_clean_motion():
    r=score(prediction('clean','C',[0,3]),gold('clean'),{})
    assert r['expected_visible']=='uncertain' and r['unsupported_clean'] and not r['full_correct']


def test_semantic_score_survives_schema_failure():
    p=prediction('clean','D');v=json.loads(p['raw']);v['joint_id']='joint_1';p['raw']=json.dumps(v)
    r=score(p,gold('clean'),{})
    assert r['class_correct'] and not r['protocol_valid']
