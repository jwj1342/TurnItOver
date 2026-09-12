"""Run the prepared four-condition pilot with the configured frozen generator."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing
from turnitover.models.client import complete
from turnitover.models.config import model_config,read_environment
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.repair.environment import CONDITIONS,RepairEnvironment
from turnitover.repair.loop import run_repair


def validate_bundle(prepared):
    manifest=json.loads((prepared/'manifest.json').read_text())
    if manifest.get('version')!=2 or manifest.get('status')!='prepared' or not manifest.get('files'):
        raise ValueError('Version 2 prepared bundle required; regenerate legacy preparations')
    for relative,expected in manifest['files'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe bundle path')
        if hashlib.sha256((prepared/path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Prepared input changed: {relative}')
    if manifest['maximum_generator_calls']!=len(manifest['cases'])*len(CONDITIONS)*manifest['rounds']:
        raise ValueError('Call budget inconsistent')
    return manifest


def run(prepared,output,generator,model_metadata):
    manifest=validate_bundle(prepared)
    cfg=load_generate_config(prepared/'config.yaml')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='incomplete',scope='prepared_four_condition_repair_pilot',model=model_metadata,
                prepared_manifest_sha256=hashlib.sha256((prepared/'manifest.json').read_bytes()).hexdigest(),
                maximum_generator_calls=manifest['maximum_generator_calls'],episodes=[],
                limitations=manifest['limitations'][1:])
    try:
        with ObservationSession(cfg.render,load_views(cfg.views_path)) as session:
            for case in manifest['cases']:
                uid=case['id'];folder=prepared/uid
                data=json.loads((folder/'case.private.json').read_text())
                environment=RepairEnvironment(session,ObjectProgram((folder/'reference.private.ts').read_text()),
                    Framing(**data['framing']),cfg,data['annotation'])
                initial=(folder/'initial.ts').read_text()
                for condition in CONDITIONS:
                    result=run_repair(initial,generator,environment.observer(condition),environment.audit,
                        output/uid/condition,max_rounds=manifest['rounds'],expected_first_request=folder/condition)
                    report['episodes'].append(dict(id=uid,condition=condition,
                        **{k:result[k] for k in ('final_passed','first_success_round','model_calls','observation_cost',
                            'non_improving_rate','exact_state_revisit_rate','stop_error_rounds','total_tokens')}))
                    print(json.dumps(report['episodes'][-1]),flush=True)
                    (output/'summary.json').write_text(json.dumps(report,indent=2))
        grouped=defaultdict(list)
        for episode in report['episodes']:grouped[episode['condition']].append(episode)
        report.update(status='complete',conditions={condition:dict(n=len(rows),
            final_passed=sum(r['final_passed'] for r in rows),model_calls=sum(r['model_calls'] for r in rows),
            total_tokens=sum(r['total_tokens'] for r in rows),observation_cost=sum(r['observation_cost'] for r in rows))
            for condition,rows in grouped.items()})
    except Exception as exc:
        report['error_type']=type(exc).__name__;raise
    finally:(output/'summary.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--env-file',type=Path,default=Path('.env'))
    args=parser.parse_args()
    validate_bundle(args.prepared)
    config=model_config('generator',read_environment(args.env_file))
    config.validate()  # Fail before creating output or calling a model when configuration is absent.
    print(json.dumps(run(args.prepared,args.out,lambda prompt,images:complete(config,prompt,images),config.public()),indent=2))
