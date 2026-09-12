"""Online zero-shot action selection with post-hoc counterfactual exposure scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

from turnitover.assets.source import make_source
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.dataset.prepare import records
from turnitover.models.local_qwen import LocalQwen
from turnitover.oracle.evidence import ExposureThresholds, exposure_matrix
from turnitover.oracle.evidence import replay_corruptions
from turnitover.policy.loop import execute_observation
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.verifier.contracts import Context
from turnitover.verifier.loop import run_verification
from turnitover.verifier.policies import VisionPolicy


def replay_images(session, program, framing, history):
    """Replay all actions including state restorations; runtime-only actions have no image."""
    images=[]
    session.load(program, framing=framing)
    try:
        for obs in history:
            def sink(step, png):
                images.append(png)
                return str(step)
            execute_observation(session, obs.action, obs.step, sink)
    finally:
        session.dispose()
    return images


def run(oracle, config, model_path, output, limit=None):
    manifest=json.loads((oracle/'manifest.json').read_text())
    if manifest['status']!='complete': raise ValueError('Complete oracle required')
    benchmark=Path(manifest['benchmark'])
    cfg=load_generate_config(config)
    source=make_source(**cfg.asset_source)
    inputs={r['id']:r for r in map(json.loads,(benchmark/'prepared/test.inputs.jsonl').read_text().splitlines())}
    raw={hashlib.sha256(f'{s.source_id}:{s.sample_id}'.encode()).hexdigest()[:24]:s
         for s,_ in records(sorted((benchmark/'raw').glob('*.tar')))}
    selected=list(inputs)[:limit] if limit is not None else list(inputs)
    output.mkdir(parents=True,exist_ok=False)
    model=LocalQwen(model_path)
    report=dict(status='incomplete',model=str(model_path),budget=manifest['budget'],
                condition='single_reference_online_active',samples=len(selected),
                limitations=['Zero-shot active policy; no fine-tuning.',
                             'Exposure is a post-hoc pixel proxy and never enters model prompts.',
                             'Failed or early-terminated episodes retain their actual observation cost.'])
    totals=Counter(); classes={}; episodes=[]
    try:
        with ObservationSession(cfg.render,load_views(cfg.views_path)) as session:
            for uid in selected:
                sample=raw[uid]
                reference=ObjectProgram.from_spec(source.get(sample.asset_id))
                candidate=replay_corruptions(reference,sample.corruptions)
                if reference.sha!=sample.reference_sha or candidate.sha!=sample.program_sha:
                    raise ValueError('Stale source or replay mismatch')
                folder=output/uid
                (folder/'observations').mkdir(parents=True)
                shutil.copyfile(oracle/'images'/uid/'reference.png',folder/'reference.png')
                def sink(step,png):
                    relative=Path('observations')/f'{step:03d}.png'
                    (folder/relative).write_bytes(png)
                    return str(relative)
                policy=VisionPolicy(model,folder,[folder/'reference.png'],mode='active',seed=sample.seed)
                session.load(candidate,framing=sample.framing)
                try:
                    episode=run_verification(session,policy,Context(**inputs[uid]['context']),manifest['budget'],sink)
                finally:
                    session.dispose()
                (folder/'episode.json').write_text(json.dumps(to_dict(episode),indent=2))
                visual=[o for o in episode.trajectory if o.image_ref]
                cand_images=[(folder/o.image_ref).read_bytes() for o in visual]
                # Teacher computation happens only after the complete learner episode.
                if visual:
                    ref_images=replay_images(session,reference,sample.framing,episode.trajectory)
                    repaired=[replay_images(session,replay_corruptions(reference,sample.corruptions,i),sample.framing,episode.trajectory)
                              for i in range(len(sample.corruptions))]
                    exposure,signals=exposure_matrix(cand_images,ref_images,repaired,ExposureThresholds(**manifest['thresholds']))
                    covered=set().union(*map(set,exposure))
                else:
                    exposure,signals,covered=[],{},set()
                expected={c.defect_id for c in sample.corruptions}
                predicted={f.defect_id for f in episode.verdict.findings}
                for label in expected|predicted:
                    counts=classes.setdefault(label,Counter())
                    counts['tp' if label in expected&predicted else 'fp' if label in predicted else 'fn']+=1
                totals['labels']+=len(sample.corruptions); totals['exposed']+=len(covered)
                totals['spent']+=episode.spent; totals[episode.termination]+=1
                record=dict(id=uid,termination=episode.termination,spent=episode.spent,status=episode.verdict.status,
                            expected=sorted(expected),predicted=sorted(predicted),exposed_labels=sorted(covered),
                            visual_steps=[o.step for o in visual],exposure=exposure,signals=signals)
                (folder/'evaluation.private.json').write_text(json.dumps(record,indent=2))
                episodes.append({k:v for k,v in record.items() if k not in ('signals','exposure')})
                print(json.dumps(episodes[-1]),flush=True)
        report.update(status='complete',totals=dict(totals),classes=classes,episodes=episodes,
                      exposure_recall=totals['exposed']/totals['labels'] if totals['labels'] else None)
    except Exception as exc:
        report.update(error=str(exc),totals=dict(totals),episodes=episodes)
        raise
    finally:
        (output/'metrics.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',type=Path,default=Path('data/benchmarks/oracle-v1'))
    parser.add_argument('--config',type=Path,default=Path('configs/generate_replicacad.yaml'))
    parser.add_argument('--model',type=Path,default=Path('models/Qwen3-VL-2B-Instruct'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    print(json.dumps(run(args.oracle,args.config,args.model,args.out,args.limit),indent=2))
