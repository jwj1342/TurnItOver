"""Browser-backed repair audit and explicit feedback conditions for prepared cases."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from turnitover.checkers.evidence import collect_evidence, resolve_states
from turnitover.checkers.geometry import GeometryAlignmentChecker
from turnitover.checkers.runtime_budget import RuntimeBudgetChecker
from turnitover.config import parse_action
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.policy.loop import execute_observation
from turnitover.render.session import HarnessError
from turnitover.render.transpile import CompileError
from turnitover.repair.loop import Feedback

CONDITIONS=('fixed','gold_constraints','oracle_observations','combined')


class RepairEnvironment:
    def __init__(self,session,reference,framing,config,annotation):
        self.session,self.reference,self.framing,self.annotation=session,reference,framing,annotation
        info=session.load(reference,framing=framing)
        self.joints=set(info.joints)
        self.states=resolve_states(config.evidence_states,tuple(info.joints))
        self.gold=collect_evidence(session,reference.sha,self.states)
        session.dispose()
        self.checkers=(GeometryAlignmentChecker(**config.checkers['geometry.alignment']),
                       RuntimeBudgetChecker(**config.checkers['runtime.budget']))

    def audit(self,source):
        program=ObjectProgram(source)
        try:
            info=self.session.load(program,framing=self.framing)
            if not self.joints.issubset(info.joints):
                return dict(usable=False,passed=False,score=0.,error='Required task joints removed')
            evidence=collect_evidence(self.session,program.sha,self.states)
            checks=[checker.check(evidence,self.gold if checker.needs_reference else None) for checker in self.checkers]
            invariants=[i for check in checks for i in check.invariants]
            return dict(usable=True,passed=all(c.passed for c in checks),
                        score=sum(i.passed for i in invariants)/len(invariants),
                        checks=[to_dict(c) for c in checks],scope='configured_states_geometry_and_runtime')
        except (CompileError,HarnessError) as exc:
            return dict(usable=False,passed=False,score=0.,error=type(exc).__name__)
        finally:self.session.dispose()

    def observer(self,condition):
        if condition not in CONDITIONS: raise ValueError('Unknown feedback condition')
        def observe(source,gold,folder):
            # This pilot deliberately freezes the initial corruption's witness recipes.
            mode='oracle' if condition in ('oracle_observations','combined') else 'fixed'
            trajectory=self.annotation['trajectories'][mode]['history']
            images=[]; observations=[]
            self.session.load(self.reference,framing=self.framing)
            try:
                path=folder/'reference.png'
                path.write_bytes(self.session.request_view('front').png)
                images.append(path)
            finally:self.session.dispose()
            self.session.load(ObjectProgram(source),framing=self.framing)
            try:
                for record in trajectory:
                    action=parse_action(record['action'])
                    def sink(step,png):
                        path=folder/f'candidate-{step:03d}.png';path.write_bytes(png);images.append(path)
                        return path.name
                    obs=execute_observation(self.session,action,record['step'],sink)
                    observations.append(dict(step=obs.step,action=record['action'],image_ref=obs.image_ref,
                                             payload={k:v for k,v in obs.payload.items() if k!='render_ms'}))
            finally:self.session.dispose()
            public=dict(task='Repair deviations from the reference while preserving task joint identities.',
                        observations=observations,reference_condition='single_front_rest_reference')
            if condition in ('gold_constraints','combined'):
                public['gold_constraint_feedback']=dict(passed=gold['passed'],
                    violations=[i for check in gold.get('checks',[]) for i in check['invariants'] if not i['passed']],
                    scope=gold.get('scope'))
            return Feedback(public,tuple(images),len(observations))
        return observe
