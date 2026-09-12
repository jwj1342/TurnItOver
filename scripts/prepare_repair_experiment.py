"""Prepare reviewable generator inputs for four matched feedback conditions; no API calls."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from turnitover.assets.source import make_source
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.dataset.prepare import records
from turnitover.models.config import model_config,read_environment
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.repair.environment import CONDITIONS,RepairEnvironment
from turnitover.repair.loop import prepare_request


def prepare(oracle,config,output,rounds=3):
    manifest=json.loads((oracle/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Complete source required')
    cfg=load_generate_config(config)
    source=make_source(**cfg.asset_source)
    benchmark=Path(manifest['benchmark'])
    samples={hashlib.sha256(f'{s.source_id}:{s.sample_id}'.encode()).hexdigest()[:24]:s
             for s,_ in records(sorted((benchmark/'raw').glob('*.tar')))}
    selected={}
    for line in (oracle/'annotations.private.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['split']!='test':continue
        key=tuple(sorted(c['defect_id'] for c in row['labels']))
        selected.setdefault(key,row)
    if len(selected)!=4:raise ValueError('Expected clean, offset, axis, and combined pilot strata')
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.yaml').write_bytes(config.read_bytes())
    report=dict(version=2,status='incomplete',task='prepared_four_condition_repair_pilot',cases=[],rounds=rounds,
                maximum_generator_calls=len(selected)*len(CONDITIONS)*rounds,
                model_config=model_config('generator',read_environment(Path('.env'))).public(),
                limitations=['No generator calls made during preparation.',
                             'Observation recipes are frozen from the initial corruption oracle, not recomputed after edits.',
                             'Gold feedback describes failed deterministic constraints, not perfect semantic diagnosis.',
                             'This pilot is not the complete RP three-oracle upper-bound experiment.'])
    try:
        with ObservationSession(cfg.render,load_views(cfg.views_path)) as session:
            for key,row in selected.items():
                uid=row['id'];sample=samples[uid]
                reference=ObjectProgram.from_spec(source.get(sample.asset_id))
                if reference.sha!=sample.reference_sha:raise ValueError('Stale asset')
                environment=RepairEnvironment(session,reference,sample.framing,cfg,row)
                gold=environment.audit(sample.program_source)
                folder=output/uid;folder.mkdir()
                (folder/'initial.ts').write_text(sample.program_source)
                (folder/'reference.private.ts').write_text(reference.source)
                (folder/'case.private.json').write_text(json.dumps(dict(framing=asdict(sample.framing),annotation=row)))
                (folder/'initial.audit.private.json').write_text(json.dumps(gold,indent=2))
                for condition in CONDITIONS:
                    target=folder/condition;target.mkdir()
                    packet=environment.observer(condition)(sample.program_source,gold,target)
                    _,paths=prepare_request(target,sample.program_source,packet,[],rounds)
                    (target/'images.json').write_text(json.dumps([p.name for p in paths]))
                report['cases'].append(dict(id=uid,stratum=list(key),initial_passed=gold['passed'],initial_score=gold['score']))
        report['status']='prepared'
        report['files']={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(output.rglob('*')) if p.is_file()}
    except Exception as exc:
        report['error_type']=type(exc).__name__;raise
    finally:(output/'manifest.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',type=Path,default=Path('data/benchmarks/oracle-v1'))
    parser.add_argument('--config',type=Path,default=Path('configs/generate_replicacad.yaml'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=prepare(args.oracle,args.config,args.out)
    print(json.dumps({**{k:v for k,v in report.items() if k!='files'},'hashed_files':len(report.get('files',{}))},indent=2))
