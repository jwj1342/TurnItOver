import json

import pytest

from turnitover.models.client import ModelError, ModelResult
from turnitover.repair.loop import Feedback, run_repair
from turnitover.repair.patches import apply_response, elide, expand

GEOMETRY = '['+', '.join(f'{i*0.25:.2f}' for i in range(80))+']'
OTHER = '['+', '.join(f'{-i*1.5e-3:.6f}' for i in range(40))+']'


def test_patches_are_atomic_unique_and_nonoverlapping():
    assert apply_response('a b',{'edits':[{'old':'a','new':'b'},{'old':'b','new':'c'}]})=='b c'
    assert apply_response('x',{'stop':True}) is None
    for source,response in [('aaa',{'edits':[{'old':'aa','new':'b'}]}),
                            ('abc',{'edits':[{'old':'ab','new':'x'},{'old':'bc','new':'y'}]}),
                            ('a',{'stop':1}),('a',{'edits':[{'old':'a','new':'a'}]})]:
        with pytest.raises(ValueError): apply_response(source,response)


def test_loop_keeps_regressions_and_counts_failed_iterations_without_leaking_gold(tmp_path):
    outputs=iter([{'edits':[{'old':'bad','new':'good'}]},
                  {'edits':[{'old':'good','new':'bad'}]},
                  {'edits':[{'old':'not present','new':'x'}]}])
    def generator(prompt,images):
        assert 'PRIVATE_GOLD' not in prompt
        assert 'before_score' not in prompt
        return ModelResult(json.dumps(next(outputs)),'fake',{'input_tokens':10,'output_tokens':10},'completed',0.)
    def audit(source): return dict(usable=True,passed=source=='good',score=float(source=='good'),secret='PRIVATE_GOLD')
    def observe(source,gold,folder): return Feedback({'task':'repair'},(),2)
    result=run_repair('bad',generator,observe,audit,tmp_path/'run')
    assert result['score_trace']==[0.,1.,0.,0.]
    assert result['first_success_round']==1
    assert not result['final_passed']
    assert result['applied_patches']==2
    assert result['exact_state_revisit_rate']==pytest.approx(1/2)
    assert result['non_improving_rate']==pytest.approx(1/2)
    assert result['stop_error_rounds']==2
    assert result['total_tokens']==60
    assert result['observation_cost']==6


def test_audit_failure_is_not_mislabeled_as_invalid_patch(tmp_path):
    def audit(source):
        if source=='new': raise ValueError('audit failure')
        return dict(usable=True,passed=False,score=0.)
    def generator(prompt,images):return ModelResult('{"edits":[{"old":"old","new":"new"}]}','fake',{},'completed',0.)
    with pytest.raises(ValueError,match='audit failure'):
        run_repair('old',generator,lambda *args:Feedback({},(),0),audit,tmp_path/'run')
    assert json.loads((tmp_path/'run/result.json').read_text())['status']=='incomplete'


def test_prepared_request_image_references_match_actual_roster(tmp_path):
    from turnitover.repair.loop import prepare_request
    reference=tmp_path/'reference.png';reference.write_bytes(b'reference')
    candidate=tmp_path/'candidate.png';candidate.write_bytes(b'candidate')
    packet=Feedback({'observations':[{'image_ref':'candidate.png','step':0}]},(reference,candidate),1)
    prompt,paths=prepare_request(tmp_path,'source',packet,[],3)
    payload=json.loads(prompt.splitlines()[-1])
    assert payload['feedback']['observations'][0]['image_ref']==payload['image_order'][1]
    assert paths[1].read_bytes()==b'candidate'
    assert packet.public['observations'][0]['image_ref']=='candidate.png'


def test_bundle_mutation_rejected_before_execution(tmp_path):
    import hashlib
    from scripts.run_repair_experiment import validate_bundle
    file=tmp_path/'initial.ts';file.write_text('initial')
    manifest=dict(version=2,status='prepared',cases=[{'id':'case'}],rounds=3,maximum_generator_calls=12,
                  files={'initial.ts':hashlib.sha256(file.read_bytes()).hexdigest()})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert validate_bundle(tmp_path)['version']==2
    file.write_text('changed')
    with pytest.raises(ValueError,match='Prepared input changed'):validate_bundle(tmp_path)


def test_token_cost_counts_router_and_provider_reasoning_usage():
    from turnitover.repair.loop import usage_tokens
    assert usage_tokens({'prompt_tokens':100,'completion_tokens':30,'total_tokens':130})==130
    assert usage_tokens({'input_tokens':100,'output_tokens':30})==130
    assert usage_tokens({'promptTokenCount':100,'candidatesTokenCount':20,'thoughtsTokenCount':10})==130
    assert usage_tokens({'promptTokenCount':100,'candidatesTokenCount':20,'totalTokenCount':130})==130


def test_elision_round_trips_and_keeps_placeholders_fixed():
    source=f'const a = {GEOMETRY};\nconst b = [1, 2, 3];\nconst c = {OTHER};\n'
    elided,arrays=elide(source)
    assert elided=='const a = __GEOM_0__;\nconst b = [1, 2, 3];\nconst c = __GEOM_1__;\n'
    assert arrays==[GEOMETRY,OTHER] and expand(elided,arrays)==source
    edited=apply_response(elided,{'edits':[{'old':'const b = [1, 2, 3]','new':'const b = [1, 2, 4]'}]})
    assert expand(edited,arrays)==source.replace('[1, 2, 3]','[1, 2, 4]')
    for edit in [{'old':'a = __GEOM_0__','new':'a = []'},                     # contains and removes
                 {'old':'GEOM_1','new':'GEOM_0'},                              # partial overlap
                 {'old':'__;\nconst b','new':'__;\nlet b'},                     # partial overlap at the end
                 {'old':'const b = [1, 2, 3]','new':'const b = __GEOM_1__'},   # duplicates
                 {'old':'const b','new':'const b = __GEOM_'}]:                 # writes placeholder text
        with pytest.raises(ValueError,match='placeholders'): apply_response(elided,{'edits':[edit]})


def test_loop_shows_elided_program_and_restores_geometry(tmp_path):
    initial=f'const shape = {GEOMETRY};\nconst axis = [1, 0, 0];\n'
    outputs=iter([{'edits':[{'old':'__GEOM_0__','new':'[0, 0]'}]},
                  {'edits':[{'old':'axis = [1, 0, 0]','new':'axis = [0, 1, 0]'}]}])
    def generator(prompt,images):
        assert GEOMETRY not in prompt and '__GEOM_0__' in prompt
        return ModelResult(json.dumps(next(outputs)),'fake',{},'stop',0.)
    audited=[]
    def audit(source):
        audited.append(source)
        return dict(usable=True,passed='[0, 1, 0]' in source,score=float('[0, 1, 0]' in source))
    result=run_repair(initial,generator,lambda *args:Feedback({},(),0),audit,tmp_path/'run',max_rounds=2)
    assert [r['outcome'] for r in result['rounds']]==['invalid_patch','applied']
    assert audited[-1]==initial.replace('[1, 0, 0]','[0, 1, 0]') and result['final_passed']
    assert result['rounds'][0]['program_chars']==len(initial)
    assert result['rounds'][0]['elided_program_chars']==len(elide(initial)[0])


@pytest.mark.parametrize('failure',['raise','truncate'])
def test_generator_error_ends_only_that_episode(tmp_path,failure):
    def generator(prompt,images):
        if failure=='raise': raise ModelError('openai_compatible request failed with HTTP 400')
        return ModelResult('{"edits": [','fake',{},'length',0.)
    result=run_repair('old',generator,lambda *args:Feedback({},(),0),
                      lambda source:dict(usable=True,passed=False,score=0.),tmp_path/'run')
    saved=json.loads((tmp_path/'run/result.json').read_text())
    assert result['status']==saved['status']=='generator_error'
    assert ('HTTP 400' if failure=='raise' else 'length') in saved['error']
