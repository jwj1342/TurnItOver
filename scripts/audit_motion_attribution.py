"""Independent rigid-motion check of browser-exported vertices, plus input isolation audit."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from scripts.build_motion_attribution import SOURCES, dump
from scripts.build_visual_decisions import sha
from scripts.evaluate_modular_verifier import _read_jsonl
from turnitover.assets.source import make_source
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.dataset.prepare import records
from turnitover.oracle.evidence import replay_corruptions
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
import hashlib


def rigid_fit(a,b):
    """Estimate a world-space rigid transform from corresponding vertices, no spec axes."""
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    ac,bc=a.mean(0),b.mean(0)
    u,_,vt=np.linalg.svd((a-ac).T@(b-bc))
    fix=np.eye(3);fix[-1,-1]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@fix@u.T;translation=bc-rotation@ac
    residual=float(np.max(np.linalg.norm(a@rotation.T+translation-b,axis=1)))
    return rotation,translation,residual


def rotation_distance(a,b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))


def main(root):
    bench=Path('../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1').resolve()
    cfg=load_generate_config(Path('configs/generate_partnet_mobility_supervision.yaml'),repo_root=bench.parents[2])
    harness=Path('../TurnItOver').resolve()
    cfg=replace(cfg,render=replace(cfg.render,harness_dir=harness/'web/dist',esbuild_bin=harness/'web/node_modules/.bin/esbuild'))
    source=make_source(**cfg.asset_source)
    raw={hashlib.sha256(f'{s.source_id}:{s.sample_id}'.encode()).hexdigest()[:24]:s for s,_ in records(sorted((bench/'raw').glob('*.tar')))}
    result=[]
    with ObservationSession(cfg.render,load_views(cfg.views_path)) as session:
        for uid in SOURCES:
            sample=raw[uid];ref=ObjectProgram.from_spec(source.get(sample.asset_id))
            axis=next(c for c in sample.corruptions if c.defect_id=='kinematics.joint_axis')
            offset=next(c for c in sample.corruptions if c.defect_id=='structure.part_offset')
            joint=ref.spec.joint(axis.params['joint']);end=joint.limits[1] or joint.limits[0]
            transforms={};world={}
            for kind,labels in [('reference',[]),('axis',[axis]),('offset',[offset]),('clean',[])]:
                session.load(replay_corruptions(ref,labels),framing=sample.framing)
                poses=[]
                for q in (0.,end):
                    session.reset_joints();session.set_joints({joint.id:q});poses.append(session.export_geometry())
                session.dispose();world[kind]=poses
                transforms[kind]={p:rigid_fit(poses[0][p].vertices,poses[1][p].vertices) for p in poses[0] if len(poses[0][p].vertices)>=3}
            for kind in ('axis','offset','clean'):
                distances={p:rotation_distance(r[0],transforms['reference'][p][0]) for p,r in transforms[kind].items()}
                rest={p:float(np.max(np.linalg.norm(world[kind][0][p].vertices-world['reference'][0][p].vertices,axis=1))) for p in transforms[kind]}
                # Class inferred only from observed geometric changes, not corruption label.
                inferred='axis' if max(distances.values())>1 else 'offset' if max(rest.values())>.001 else 'clean'
                assert inferred==kind
                row={'id':sample.asset_id+'-'+kind,'inferred_class':inferred,'rotation_difference_deg':distances,
                     'rest_max_vertex_delta_m':rest,'rigid_fit_max_residual_m':max(r[2] for r in transforms[kind].values())}
                result.append(row)
    inputs=_read_jsonl(root/'inputs.jsonl');triplets={}
    for r in inputs:
        if r['condition'] not in ('B','C'):continue
        triplets.setdefault((r['asset_id'],r['condition']),[]).append(r['messages'][0]['content'])
        assert r['messages'][1]['content']==''
    assert all(len(v)==3 and len(set(v))==1 for v in triplets.values())
    historical=json.loads((root/'historical.audit.json').read_text())
    replay={r['id']:r for r in _read_jsonl(root/'v4-historical/predictions.jsonl')}
    match=[{'id':r['id'],'raw_identical':r['original_raw']==replay[r['id']]['raw'],
            'token_count_identical':r['original_input_tokens']==replay[r['id']]['input_tokens']} for r in historical]
    dump(root/'geometry.audit.json',{'geometry':result,'B_C_triplet_text_identical':True,
          'historical_replay':match,'method':'SVD rigid registration of browser-exported world vertices at rest/end; no axis vectors used in fitting',
          'script_sha256':sha(Path(__file__))})
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
