import json
from types import SimpleNamespace

from scripts.score_kimi_v4_increment import score


def test_strict_scoring_does_not_treat_partial_evidence_as_exact(tmp_path):
    context = {'task':'inspect', 'parts':['door'], 'joints':['hinge'], 'views':['front'],
               'runtime_properties':[], 'action_types':[], 'max_triangles':1, 'max_draw_calls':1}
    finding = {'defect_id':'kinematics.joint_axis', 'joint_id':'hinge', 'parts':['door'],
               'severity':.8, 'confidence':.8, 'evidence_steps':[0,1],
               'description':'Motion differs.', 'suggested_fix':'Correct axis.'}
    target = {'verdict':{'status':'fail', 'confidence':.8, 'findings':[finding],
                         'summary':'Wrong axis.', 'limitations':[]}}
    payload = {'context':context, 'history':[{'step':0},{'step':1}], 'remaining':0}
    record = {'id':'a:verdict','episode_id':'a','terminal':True,'data_identity':{'asset_id':'asset'},
              'messages':[{'role':'user','content':'protocol\n'+json.dumps(payload)},
                          {'role':'assistant','content':json.dumps(target)}]}
    (tmp_path/'validation.episodes.jsonl').write_text(json.dumps(record)+'\n')
    annotation = {'id':'a','split':'validation','labels':[{'defect_id':'kinematics.joint_axis'}]}
    (tmp_path/'annotations.private.jsonl').write_text(json.dumps(annotation)+'\n')
    for arm in ('baseline','experiment'):
        (tmp_path/arm).mkdir()
        prediction = json.loads(json.dumps(target))
        if arm == 'baseline':
            prediction['verdict']['findings'][0]['evidence_steps']=[0]
        row = {'id':'a:verdict','raw':json.dumps(prediction),'evidence_consistent':True}
        (tmp_path/arm/'predictions.jsonl').write_text(json.dumps(row)+'\n')
    score(SimpleNamespace(dataset=tmp_path, baseline=tmp_path/'baseline',
                          experiment=tmp_path/'experiment',out=tmp_path/'score.json'))
    result=json.loads((tmp_path/'score.json').read_text())
    baseline=result['arms']['baseline']['all']
    assert baseline['exact_verdict']==baseline['evidence_supported']==1
    assert baseline['strict_all']==baseline['evidence']==0
    assert result['arms']['experiment']['all']['strict_all']==1
    assert result['paired'][0]['strict_all_change']==1
