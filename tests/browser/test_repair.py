import json
from pathlib import Path

import pytest

from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.models.client import ModelResult
from turnitover.repair.environment import RepairEnvironment
from turnitover.repair.loop import run_repair

pytestmark=pytest.mark.browser


def test_real_rendered_repair_and_feedback_isolation(session_factory,toy_spec,tmp_path):
    # Upper detent is rest: audit must also inspect lower/mid to find axis corruption.
    spec=toy_spec.replace_joint('door',limits=(-1.,0.))
    reference=ObjectProgram.from_spec(spec)
    candidate=ObjectProgram.from_spec(spec.replace_joint('door',axis=(1.,0.,0.)))
    cfg=load_generate_config(Path('configs/generate_replicacad.yaml'))
    info=session_factory.load(reference,framing=None)
    session_factory.dispose()
    history=[dict(step=0,action=dict(type='request_view',view_id='front')),
             dict(step=1,action=dict(type='actuate_joint',joint_id='door',detent='zero'))]
    annotation=dict(trajectories={mode:dict(history=history) for mode in ('fixed','oracle')})
    environment=RepairEnvironment(session_factory,reference,info.framing,cfg,annotation)
    broken=environment.audit(candidate.source)
    assert broken['usable'] and not broken['passed']
    assert any(not i['passed'] and i['state']=='joint:door:zero' for c in broken['checks'] for i in c['invariants'])
    fixed_folder=tmp_path/'fixed_feedback';fixed_folder.mkdir()
    gold_folder=tmp_path/'gold_feedback';gold_folder.mkdir()
    normal=environment.observer('fixed')(candidate.source,broken,fixed_folder)
    oracle=environment.observer('gold_constraints')(candidate.source,broken,gold_folder)
    assert 'gold_constraint_feedback' not in normal.public
    assert oracle.public['gold_constraint_feedback']['violations']
    assert normal.observation_cost==oracle.observation_cost==2
    assert [p.read_bytes() for p in normal.images]==[p.read_bytes() for p in oracle.images]
    outputs=iter([dict(edits=[dict(old=candidate.source,new=reference.source)]),dict(stop=True)])
    def generator(prompt,images):
        assert 'gold_constraint_feedback' not in prompt
        return ModelResult(json.dumps(next(outputs)),'fixture-not-a-model',{},'completed',0.)
    result=run_repair(candidate.source,generator,environment.observer('fixed'),environment.audit,tmp_path/'run')
    assert result['final_passed']
    assert result['first_success_round']==1
    assert result['termination']=='generator_stop'
    assert result['model_calls']==2
    assert result['observation_cost']==4
    assert (tmp_path/'run/final.ts').read_text()==reference.source
