"""Build paired direct/intermediate supervision from frozen RGB and public states."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
from turnitover.oracle.motion_grounding import compare_observations, visible_comparison, observable_signature

INSTRUCTION = '''Images are REFERENCE,CANDIDATE pairs in the public states order. Compare matched positions
and reference rest-to-actuated versus candidate rest-to-actuated image changes before diagnosis.
The RGB-derived fractions and bounding boxes are computed from these exact images with a fixed erosion.
A temporal residual alone does NOT prove an axis defect: a displaced part can also change the moving silhouette.
In this controlled single-defect task, visible rest displacement supports offset; matching rest plus discrepant
motion can support axis, but occluded displacement remains ambiguous. Clean requires observed matching motion.
Missing motion with no visible displacement means uncertain. Only diagnose tested visible motion.
Use legal hierarchy IDs; joint is null for offset; both IDs null for clean/uncertain.
Evidence steps must include both rest and actuated endpoints for axis, a visible displaced state for offset,
and all observed states for clean. uncertain uses []. Return JSON with defect_type, joint, part, evidence_steps.
'''
INTERMEDIATE = '''First output comparison with static_rest (matching/different), static_steps (visibly different pair IDs),
motion_change (matching/different/unobserved), motion_pairs (rest/actuated ID pairs with different image changes).
These describe pixels, not private rotation parameters. Then output diagnosis in the same JSON object.
'''


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2))


def public_of(row):
    return json.loads(row['messages'][0]['content'].split('Public input: ',1)[1])


def hierarchy_ids(node):
    # Keep the legal ID tree, not transforms or object identity.
    name=node.get('name','')
    return {'name':name if name.startswith(('link_','joint:','base')) else 'object',
            'children':[hierarchy_ids(n) for n in node.get('children',[])]}


def make_row(row, source, root, indices, condition, variant):
    public=public_of(row); states=[public['states'][i] for i in indices]
    pairs=[]; paths=[]
    for i in indices:
        pair=[]
        for path in row['images'][2*i:2*i+2]:
            src=source/path; relative=Path('images')/sha(src)[:20]/'observation.png'; dst=root/relative
            if not dst.exists(): dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
            paths.append(str(relative)); pair.append(Image.open(dst).convert('RGB'))
        pairs.append(pair)
    features=compare_observations(pairs,states,public['rest_value'])
    comparison=visible_comparison(features,states,public['rest_value'])
    signature=observable_signature(comparison)
    target=copy.deepcopy(row['evidence_target']); latent=target['defect_type']
    if condition!='rest_only' and latent in ('axis','offset') and not comparison['static_steps']:
        target={'defect_type':'uncertain','joint':None,'part':None,'evidence_steps':[]}
    if condition=='rest_only':
        target={'defect_type':'offset' if latent=='offset' and comparison['static_rest']=='different' else 'uncertain',
                'joint':None,'part':target['part'] if latent=='offset' and comparison['static_rest']=='different' else None,'evidence_steps':[]}
    if target['defect_type']=='offset': target['evidence_steps']=comparison['static_steps'][:1]
    elif target['defect_type']=='axis': target['evidence_steps']=comparison['motion_pairs'][0] if comparison['motion_pairs'] else []
    elif target['defect_type']=='clean': target['evidence_steps']=[s['step'] for s in states]
    visible={'hierarchy':hierarchy_ids(public['hierarchy']['root']), 'joint':public['joint'],
             'states':states,'rest_value':public['rest_value'],'rgb_comparison':features}
    prompt=INSTRUCTION+(INTERMEDIATE if variant=='intermediate' else '')+'Public input: '+json.dumps(visible,sort_keys=True)
    if variant=='intermediate': target={'comparison':comparison,**target}
    result={'id':row['id']+':'+condition,'asset_id':row['asset_id'],'condition':condition,
        'images':paths,'image_sha256':[sha(root/p) for p in paths], 'messages':[{'role':'user','content':prompt}],
        'ordinary_target':target,'evidence_target':target,
        'gold':{'latent_class':latent,'comparison':comparison,'signature':signature,
                'signature_matches_gold':signature==target['defect_type'], 'source_id':row['id']}}
    return result


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    old=json.loads((args.v1/'manifest.json').read_text())
    audit=[]; result={}
    for split in ('train','validation'):
        rows=[r for r in read(args.source/f'{split}.jsonl') if r['condition']=='full_motion']
        if split=='train': rows=[r for r in rows if r['asset_id'] in old['train_assets']]
        built=[]
        for row in rows:
            public=public_of(row); all_indices=list(range(len(public['states'])))
            rest=[i for i,s in enumerate(public['states']) if abs(s['actual_value']-public['rest_value'])<1e-6]
            for condition,indices in [('full_motion',all_indices),('rest_only',rest)]:
                r=make_row(row,args.source,args.out,indices,condition,args.variant)
                built.append(r); audit.append({'split':split,'id':r['id'],**r['gold']})
        # A full balanced object is admitted only if BOTH defect signatures are observable.
        rejected={r['asset_id'] for r in built if r['condition']=='full_motion' and not r['gold']['signature_matches_gold']}
        if split=='train': built=[r for r in built if r['asset_id'] not in rejected]
        result[split]=built
    train={r['asset_id'] for r in result['train']}; val={r['asset_id'] for r in result['validation']}
    assert not train & val
    for split,rows in result.items():
        (args.out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    dump(args.out/'visual-audit.private.json',audit)
    dump(args.out/'manifest.json',{'status':'complete','formal_training':True,'version':2,'variant':args.variant,
        'train_assets':sorted(train),'validation_assets':sorted(val),'train_rows':len(result['train']),
        'validation_rows':len(result['validation']),'frozen_test_used':False,
        'source':str(args.source.resolve()),'source_manifest_sha256':sha(args.source/'manifest.json'),
        'builder_sha256':sha(__file__),'representation_sha256':sha('turnitover/oracle/motion_grounding.py'),
        'classes':{s:dict(Counter(r['evidence_target']['defect_type'] for r in rows)) for s,rows in result.items()},
        'caveat':'Pixel signature audit is restricted to single frozen corruptions, not semantic rotation or human Gold.'})
    print((args.out/'manifest.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--v1',type=Path,default=Path('output/partnet-motion-diagnosis-v1'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--variant',choices=['direct','intermediate'],required=True)
    main(p.parse_args())
