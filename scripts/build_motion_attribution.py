"""Small Validation-only aligned motion probe; private geometry never enters prompts."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scripts.build_visual_decisions import sha
from scripts.evaluate_modular_verifier import _read_jsonl, _templates
from turnitover.assets.source import make_source
from turnitover.config import load_generate_config, parse_action
from turnitover.core.program import ObjectProgram
from turnitover.dataset.prepare import records
from turnitover.oracle.evidence import replay_corruptions, pixels, pixel_signal
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.policy.loop import execute_observation
from turnitover.verifier.modular import build_single_reference_record

# Frozen, previously verified Validation corruptions. No new corruption magnitudes.
SOURCES = ('750796de0febcb6e917f9e61', 'e6a99ffe26382f42d5228974',
           'c3360599f68da7b11f4b866e', '873a775b1dedfd3be942d055')
INSTRUCTION = '''Compare reference and candidate images at exactly matched camera and joint values.
Images are ordered as reference, candidate pairs, one pair per listed step. All other
joints are at zero. Diagnose only the tested motion and visible parts, not whole-object validity.
axis means a wrong rotation direction/plane revealed across states; offset means a
part position displacement. A single rest pose cannot establish a rotation-axis defect
or clean motion. If evidence is insufficient say uncertain; do not guess.
Return JSON with keys abnormal (true/false/null), classification (axis/offset/clean/uncertain),
joint_id (string or null), part_id (string or null), evidence_steps (integer list),
explanation (describe visible changes). For offset joint_id is null. For clean both IDs
are null. Use only public IDs below. clean means no anomaly in these tested states only.
'''

def dump(path, value):
    path.write_text(json.dumps(value, indent=2))

def main(args):
    root = args.out.resolve(); root.mkdir(parents=True, exist_ok=False)
    bench = Path('../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1').resolve()
    v4 = bench / 'oracle-first-failure-v4'
    cfg = load_generate_config(Path('configs/generate_partnet_mobility_supervision.yaml'), repo_root=bench.parents[2])
    harness = Path('../TurnItOver').resolve()
    cfg = replace(cfg, render=replace(cfg.render, harness_dir=harness/'web/dist', esbuild_bin=harness/'web/node_modules/.bin/esbuild'))
    source = make_source(**cfg.asset_source)
    templates, _ = _templates(v4)
    traces = {r['id'].split(':')[0]: r for r in _read_jsonl(Path('data/partnet-motion-decisions-v5/trajectories.jsonl')) if r['split']=='validation' and r['id'].endswith('order0')}
    raw = {hashlib.sha256(f'{s.source_id}:{s.sample_id}'.encode()).hexdigest()[:24]:s for s,_ in records(sorted((bench/'raw').glob('*.tar')))}
    inputs, gold = [], []
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
        browser = session.browser_version()
        for uid in SOURCES:
            assert uid in traces
            sample = raw[uid]; ref = ObjectProgram.from_spec(source.get(sample.asset_id))
            assert ref.sha == sample.reference_sha
            assert replay_corruptions(ref,sample.corruptions).sha == sample.program_sha
            axis = next(c for c in sample.corruptions if c.defect_id=='kinematics.joint_axis')
            offset = next(c for c in sample.corruptions if c.defect_id=='structure.part_offset')
            joint = ref.spec.joint(axis.params['joint']); assert str(joint.type)=='revolute'
            # Fixed for all three variants of an object. Gold is NOT used to vary evidence per class.
            end = joint.limits[1] if joint.limits[1] != 0 else joint.limits[0]
            states = [dict(step=i, joint=joint.id, value=value, view=view) for i,(view,value) in enumerate((v,q) for v in ('front','oblique_fl') for q in (0.,end/2,end))]
            refs, geometries, cameras = [], [], []
            folder = root/sample.asset_id; folder.mkdir()
            session.load(ref,framing=sample.framing)
            for state in states:
                session.reset_joints(); session.set_joints({joint.id:state['value']})
                obs = session.request_view(state['view']); cameras.append(obs.camera)
                p=folder/f"reference-{state['step']}.png";p.write_bytes(obs.png);refs.append(p)
                geometries.append(session.export_geometry())
            session.dispose()
            for kind, labels in [('axis',[axis]),('offset',[offset]),('clean',[])]:
                case=f'{sample.asset_id}-{kind}'; dst=folder/kind;dst.mkdir()
                cand=replay_corruptions(ref,labels)
                # Parameter-difference check independent of label claims.
                changed_axes=[j.id for j in cand.spec.joints if j.axis!=ref.spec.joint(j.id).axis]
                changed_origins=[p.id for p in cand.spec.parts if p.origin!=ref.spec.part(p.id).origin]
                assert changed_axes==([joint.id] if kind=='axis' else [])
                assert changed_origins==([offset.parts[0]] if kind=='offset' else [])
                session.load(cand,framing=sample.framing)
                paths=[];signals=[];geometry=[]
                for state,rp,cam,rg in zip(states,refs,cameras,geometries,strict=True):
                    session.reset_joints();session.set_joints({joint.id:state['value']})
                    obs=session.request_view(state['view']);assert obs.camera==cam
                    p=dst/f"candidate-{state['step']}.png";p.write_bytes(obs.png);paths.append(p)
                    signals.append(pixel_signal(pixels(obs.png),pixels(rp.read_bytes())))
                    cg=session.export_geometry()
                    delta={p:float(np.linalg.norm(cg[p].vertices-rg[p].vertices,axis=1).max()) for p in rg if len(rg[p].vertices)}
                    geometry.append(delta)
                session.dispose()
                if kind=='axis':
                    assert max(geometry[0].values())<1e-6 and max(geometry[2].values())>.001
                elif kind=='offset': assert max(geometry[0].values())>.001
                else: assert all(max(g.values())<1e-6 for g in geometry)
                public={'parts':list(ref.spec.part_ids),'joints':[{'id':j.id,'part':j.part,'type':str(j.type)} for j in ref.spec.joints]}
                for cond,indices in [('B',list(range(6))),('C',[0,3])]:
                    image_paths=[str(p) for i in indices for p in (refs[i],paths[i])]
                    payload={'structure':public,'states':[dict(states[i],camera=cameras[i]) for i in indices]}
                    inputs.append({'id':case,'asset_id':sample.asset_id,'condition':cond,'images':image_paths,
                                   'messages':[{'role':'user','content':INSTRUCTION+json.dumps(payload)}, {'role':'assistant','content':''}]})
                # A recreates the existing fixed six-step closed-loop visible input, with one reference.
                trace=traces[uid]; history=[];a_paths=[]
                session.load(cand,framing=sample.framing)
                for step,h in enumerate(trace['history']):
                    def sink(index,png):
                        p=dst/f'A-{index}.png';p.write_bytes(png);a_paths.append(str(p));return str(p)
                    obs=execute_observation(session,parse_action(h['action']),step,sink)
                    history.append({'step':step,'action':h['action'],'payload':{k:v for k,v in obs.payload.items() if k!='render_ms'}})
                session.dispose()
                a=build_single_reference_record(templates[uid],history,0)
                a.update(id=case,asset_id=sample.asset_id,condition='A',images=[str((v4/templates[uid]['images'][0]).resolve()),*a_paths])
                a['messages'][1]['content']='' # Never persist a training target as model input.
                inputs.append(a)
                entry={'id':case,'asset_id':sample.asset_id,'source_id':uid,'class':kind,'joint':joint.id if kind=='axis' else None,
                       'part':labels[0].parts[0] if labels else None,'reference_sha':ref.sha,'candidate_sha':cand.sha,
                       'original_candidate_sha':sample.program_sha,'retained_corruptions':[asdict(c) for c in labels],
                       'changed_axes':changed_axes,'changed_origins':changed_origins,'geometry_max_vertex_delta_m':geometry,
                       'pixel_signals':signals,'states':states,'cameras':cameras,
                       'construction':'retain existing axis only / existing offset only / reference; no new labels',
                       'scope':'selected revolute joint; fixed within each object triplet'}
                gold.append(entry)
                sheet=Image.new('RGB',(6*256,2*280),'white');draw=ImageDraw.Draw(sheet)
                for i,(rp,cp) in enumerate(zip(refs,paths,strict=True)):
                    for row,p in enumerate((rp,cp)):
                        sheet.paste(Image.open(p),(i*256,row*280+24));draw.text((i*256+4,row*280+4),f"{i} {states[i]['view']} q={states[i]['value']:.2f} {'ref' if row==0 else 'cand'}",fill='black')
                sheet.save(dst/'review.png')
                print(case, [round(s['changed_fraction'],3) for s in signals],flush=True)
    for row in inputs: row['image_sha256']=[sha(Path(p)) for p in row['images']]
    (root/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputs))
    (root/'gold.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in gold))
    dump(root/'manifest.json',{'status':'complete','browser':browser,'sources':SOURCES,'cases':len(gold),'objects':4,
         'scope':'Validation counterfactual triplets; A protocol recreation, not identical historical candidates',
         'inputs_sha256':sha(root/'inputs.jsonl'),'gold_sha256':sha(root/'gold.private.jsonl'),
         'builder_sha256':sha(Path(__file__)),'frozen_test_used':False})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
