"""Frozen v1.1 multi-view repair inputs, camera replay, and independent checks."""
from dataclasses import replace
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.render.session import Framing, HarnessError, ObservationSession
from turnitover.render.transpile import CompileError
from turnitover.render.views import load_views
from turnitover.repair.environment import RepairEnvironment

MANIFEST='b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup(root,batch):
    dataset=root/'three-tier-900-sft-v1.1'
    assert sha(dataset/'manifest.json')==MANIFEST
    records={r['episode_id']:r for r in map(json.loads,(dataset/'validation.episodes.jsonl').read_text().splitlines())}
    cases=read(batch/'selection.private.json')
    assert len(cases)==44
    for case in cases:
        record=records[case['case_id']]
        assert record['admission']=='eligible' and not record['is_clean']
        assert len(record['images'])=={'easy':4,'medium':6,'hard':12}[case['difficulty']]
        for role in ('reference','candidate'):
            assert sha(batch/'assets-ready'/case['programs'][role]['typescript'])==case['programs'][role]['typescript_sha256']
    cfg=load_generate_config(root/'three-tier-900-sft-v1.1-runtime/configs/generate_replicacad.yaml',repo_root=root/'three-tier-900-sft-v1.1-runtime')
    cfg=replace(cfg,render=replace(cfg.render,harness_dir=root/'repair-assets/harness',esbuild_bin=root/'repair-bin/esbuild'))
    return dataset,records,cases,cfg
def sources(batch,case):
    return tuple((batch/'assets-ready'/case['programs'][role]['typescript']).read_text(encoding='utf-8') for role in ('reference','candidate'))
def observer(session,record,case,reference,framing,pair_type):
    prefix,raw=record['messages'][0]['content'].rsplit('\n',1)
    payload=json.loads(raw)
    cells=case['observation_cells']
    step_by_image={row['image_ref']:row['step'] for row in payload['history']}
    # Both roles map to the same publicly supplied pair. The first reference
    # image can be outside history; only successful numbered steps are cited.
    steps={step_by_image[cell[f'{role}_image']]:i for i,cell in enumerate(cells)
           for role in ('reference','candidate') if cell[f'{role}_image'] in step_by_image}
    def observe(source,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        images={}
        for role,program in [('reference',reference),('candidate',ObjectProgram(source))]:
            try:
                session.load(program,framing=framing)
                for cell in cells:
                    session.reset_joints()
                    session.set_joints({case['target_joint_id']:case['state_values'][cell['state']]})
                    name=cell[f'{role}_image'];path=folder/Path(name).name
                    path.write_bytes(session.request_view(cell['view_id']).png)
                    images[name]=path
            except (CompileError,HarnessError) as exc:
                if role=='reference':raise
                return {'gate':{'passed':False,'stage':type(exc).__name__,'diagnostic':'Candidate could not be compiled or rendered.'}}
            finally:session.dispose()
        public={'images':record['images'],'messages':[{'role':'user','content':prefix+'\n'+json.dumps(payload)},{'role':'assistant','content':''}]}
        dump(folder/'joint-request.json',{'record':public,'image_paths':[str(images[n]) for n in record['images']]})
        pairs=[pair_type(f"{c['view_id']}/{c['state']}",images[c['reference_image']],images[c['candidate_image']]) for c in cells]
        dump(folder/'observation-manifest.json',{'case_id':case['case_id'],'cells':cells,
             'generator_images':[{'file':p.name,'sha256':sha(p)} for pair in pairs for p in (pair.reference,pair.candidate)],
             'joint':case['target_joint_id'],'requested_state_values':case['state_values']})
        return {'pairs':pairs,'public_record':{'request':folder/'joint-request.json','payload':payload,'steps':steps}}
    return observe
def recover_framing(session,reference,case,record,dataset,candidate=None):
    session.load(reference,framing=None)
    vertices=[]
    for state in ('rest','mid','limit'):
        session.reset_joints();session.set_joints({case['target_joint_id']:case['state_values'][state]})
        vertices.extend(mesh.vertices for mesh in session.export_geometry().values())
    points=np.concatenate(vertices);low,high=points.min(axis=0),points.max(axis=0)
    center=tuple(float(x) for x in (low+high)/2)
    session.dispose()
    cell=case['observation_cells'][0]
    frozen={name:np.asarray(Image.open(dataset/name).convert('RGB')) for name in record['images']}
    # These are the original construction scripts' declared normal-only paddings.
    paddings=(1.15,1.1,1.25,1.0,1.5) if case['difficulty']=='hard' else (1.25,1.0,1.5)
    attempts=[]
    for padding in paddings:
        norm=np.linalg.norm(high-low)
        # Reproduce the construction's scalar arithmetic, including the
        # NumPy 1/2 scalar-promotion difference, rather than accepting drift.
        radii=dict.fromkeys([float(norm)/2*1.5*padding/1.5,float(norm)/2*padding,float(norm/2*padding),float(norm/2*1.5)*padding/1.5])
        # float32 norm reduction may differ by a few ULP across NumPy/BLAS
        # implementations. Accept only a candidate that exactly replays every
        # frozen pixel, and retain the recovered radius in the audit.
        for direction in (-np.inf,np.inf):
            alternate=np.float32(norm)
            for _ in range(3):
                alternate=np.nextafter(alternate,np.float32(direction))
                radii[float(alternate)/2*1.5*padding/1.5]=None
        for radius in radii:
            framing=Framing(center=center,radius=radius)
            equal=True;errors=[]
            for role,program in [('reference',reference),('candidate',ObjectProgram(candidate))] if candidate is not None else [('reference',reference)]:
                session.load(program,framing=framing)
                for cell in case['observation_cells']:
                    session.reset_joints();session.set_joints({case['target_joint_id']:case['state_values'][cell['state']]})
                    import io
                    fresh=np.asarray(Image.open(io.BytesIO(session.request_view(cell['view_id']).png)).convert('RGB'))
                    target=frozen[cell[f'{role}_image']]
                    matches=np.array_equal(target,fresh)
                    errors.append(float(np.abs(target.astype(float)-fresh).mean()))
                    if not matches:equal=False;break
                session.dispose()
                if not equal:break
            attempts.append({'padding':padding,'radius':radius,'equal':equal,'mae':max(errors)})
            if equal:return framing,attempts
    raise ValueError('No original camera matches frozen reference: '+json.dumps(attempts))
