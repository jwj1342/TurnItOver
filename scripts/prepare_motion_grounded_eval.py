"""Frozen-image transfer and controlled order/representation ablations; no training rows."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from scripts.build_motion_grounded import make_row, read, sha, dump


def main(a):
    a.out.mkdir(parents=True,exist_ok=False); rows=[]
    def append(row,cohort):
        row['id']=cohort+':'+row['id']; row['condition']=cohort+'/'+row['condition']; rows.append(row)
    original=read(a.dataset/'validation.jsonl')
    # Copy source PNGs under a contained path; no symlink escape in training loader.
    shutil.copytree(a.dataset/'images',a.out/'images')
    for r in original:
        for mode in ('reverse','pixels_only','features_only'):
            x=copy.deepcopy(r); text,public=x['messages'][0]['content'].split('Public input: ',1); public=json.loads(public)
            if mode=='reverse':
                x['images']=[p for i in reversed(range(len(x['images'])//2)) for p in x['images'][2*i:2*i+2]]
                x['image_sha256']=[sha(a.out/p) for p in x['images']]; public['states'].reverse()
            elif mode=='pixels_only': public.pop('rgb_comparison')
            else: x['images']=[];x['image_sha256']=[]
            x['messages'][0]['content']=text+'Public input: '+json.dumps(public,sort_keys=True)
            append(x,mode)
    for cohort,source in [('v1_frozen',a.v1),('extra',a.extra)]:
        for row in read(source/'validation.jsonl'):
            if row['condition']!='full_motion': continue
            public=json.loads(row['messages'][0]['content'].split('Public input: ',1)[1])
            full=list(range(len(public['states'])))
            rest=[i for i,s in enumerate(public['states']) if abs(s['actual_value']-public['rest_value'])<1e-6]
            for condition,indices in [('full_motion',full),('rest_only',rest)]:
                append(make_row(row,source,a.out,indices,condition,a.variant),cohort)
    gold={r['id']:r for r in read(a.frozen/'gold.private.jsonl')}
    for row in read(a.frozen/'inputs.jsonl'):
        if row['condition'] not in ('B','C'):continue
        g=gold[row['id']]; public=json.loads('{"structure":'+row['messages'][0]['content'].split('\n{"structure":',1)[1])
        # Legal runtime structure is already public in the frozen protocol.
        hierarchy={'root':{'name':'object','children':[{'name':p,'children':[]} for p in public['structure']['parts']]}}
        states=[{'step':s['step'],'view':s['view'],'actual_value':s['value']} for s in public['states']]
        p={'hierarchy':hierarchy,'joint':public['states'][0]['joint'],'states':states,'rest_value':0.}
        source=copy.deepcopy(row);source['messages']=[{'role':'user','content':'Public input: '+json.dumps(p)}]
        source['evidence_target']={'defect_type':g['class'],'joint':g['joint'],'part':g['part'],'evidence_steps':[]}
        condition='rest_only' if row['condition']=='C' else 'full_motion'
        x=make_row(source,Path('/'),a.out,list(range(len(states))),condition,a.variant)
        # Frozen B/C labels follow the prior independent visible review, never current pixel signature.
        if condition=='full_motion' or g['class']=='offset':
            t=x['evidence_target'];t.update(defect_type=g['class'],joint=g['joint'],part=g['part'])
            if g['class']=='clean':t['evidence_steps']=[s['step'] for s in states]
        append(x,'frozen_'+row['condition'])
    (a.out/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (a.out/'train.jsonl').write_text('')
    dump(a.out/'manifest.json',{'status':'complete','formal_training':True,'evaluation_only':True,'rows':len(rows),
        'variant':a.variant,'builder_sha256':sha(__file__),'frozen_test_used':False})
    print(len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',choices=['direct','intermediate'],required=True)
    p.add_argument('--v1',type=Path,default=Path('output/partnet-motion-diagnosis-v1'))
    p.add_argument('--extra',type=Path,default=Path('output/motion-grounded-v2-extra-render'))
    p.add_argument('--frozen',type=Path,default=Path('output/motion-attribution-v1'))
    main(p.parse_args())
