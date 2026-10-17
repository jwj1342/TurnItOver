"""Recompute legal visual features, verify paired arms and summarize independent predictions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from PIL import Image
from scripts.build_motion_grounded import read, sha, dump
from turnitover.oracle.motion_grounding import compare_observations


def summarize(rows):
    counts=Counter(n=len(rows)); confusion=defaultdict(Counter); objects=defaultdict(list)
    for r in rows:
        g=r['expected']['defect_type'];p=(r['predicted'] or {}).get('defect_type','invalid')
        confusion[g][p]+=1;objects[r['asset_id']].append(r)
        for k in ('class_correct','full_correct','part_correct','joint_correct','evidence_correct'):
            counts[k]+=int(r[k])
        counts['unsupported']+=int(p in ('axis','offset') and not r['full_correct'])
        if g in ('axis','offset'):
            counts['defect_n']+=1;counts['defect_part_correct']+=int(r['part_correct'])
        if g=='axis':counts['axis_n']+=1;counts['axis_joint_correct']+=int(r['joint_correct'])
        for k,v in r.get('comparison_scores',{}).items():counts['comparison_'+k]+=int(v)
    triples=0;collapsed=0;all_correct=0
    for obj,values in objects.items():
        full=[r for r in values if r['condition'].endswith('full_motion')]
        if len(full)==3:
            triples+=1;collapsed+=int(len({(r['predicted'] or {}).get('defect_type') for r in full})==1)
            all_correct+=int(all(r['class_correct'] for r in full))
    return {**counts,'confusion':{k:dict(v) for k,v in confusion.items()},
        'triplets':triples,'collapsed_triplets':collapsed,'correct_triplets':all_correct,
        'per_object':{k:{'n':len(v),'class_correct':sum(x['class_correct'] for x in v),
                       'full_correct':sum(x['full_correct'] for x in v)} for k,v in objects.items()},
        'errors':[r['id'] for r in rows if not r['full_correct']]}


def main(a):
    if a.out.exists():raise ValueError('Fresh audit required')
    r=Path('output'); left=r/'motion-grounded-v2-intermediate';right=r/'motion-grounded-v2-direct'
    checks=[]
    for split in ('train','validation'):
        x,y=read(left/f'{split}.jsonl'),read(right/f'{split}.jsonl');assert len(x)==len(y)
        for u,v in zip(x,y,strict=True):
            assert u['id']==v['id'] and u['image_sha256']==v['image_sha256']
            assert {k:val for k,val in u['evidence_target'].items() if k!='comparison'}==v['evidence_target']
            public=json.loads(u['messages'][0]['content'].split('Public input: ',1)[1])
            assert public==json.loads(v['messages'][0]['content'].split('Public input: ',1)[1])
            assert not any(key in json.dumps(public) for key in ('candidate_sha','latent_class','original_axis','severity','exposure','position','rotation'))
            images=[]
            for p,h in zip(u['images'],u['image_sha256'],strict=True):
                assert sha(left/p)==h;images.append(Image.open(left/p).convert('RGB'))
            features=compare_observations(list(zip(images[::2],images[1::2],strict=True)),public['states'],public['rest_value'])
            assert features==public['rgb_comparison']
        checks.append({'split':split,'paired_rows':len(x),'identical_images_features_labels':True,'pixel_recomputation':True})
    train={x['asset_id'] for x in read(left/'train.jsonl')}
    frozen={x['asset_id'] for x in read(r/'motion-attribution-v1/gold.private.jsonl')}
    oldval={x['asset_id'] for x in read(r/'partnet-motion-diagnosis-v1/validation.jsonl')}
    assert not train & (frozen|oldval|{'PM_10797','PM_45759'})
    results={}
    for name in a.evaluations:
        path=r/name
        if not (path/'metrics.json').exists():raise ValueError(f'Incomplete evaluation {name}')
        values=read(path/'predictions.jsonl');cohorts=defaultdict(list)
        for row in values:cohorts[row['condition'].split('/')[0] if '/' in row['condition'] else 'main'].append(row)
        results[name]={}
        for c,rows in cohorts.items():
            results[name][c]=summarize(rows)
            for condition in ('full_motion','rest_only'):
                subset=[x for x in rows if x['condition'].endswith(condition)]
                if subset:results[name][c+'/'+condition]=summarize(subset)
        results[name]['predictions_sha256']=sha(path/'predictions.jsonl')
    dump(a.out,{'status':'passed','checks':checks,'train_assets':sorted(train),'heldout_no_overlap':True,
        'independent_new_assets':['PM_10797','PM_45759'],'new_axis_assets':0,
        'results':results,'audit_sha256':sha(__file__),
        'limitations':['Pixel comparison targets are deterministic measurements, not independently annotated motion semantics.',
        'Scalar rest mismatch versus temporal residual is not a 3D rotation estimator; occluded offsets can be confounded.',
        'Single seed; seven validation objects total; new two objects have offset/clean only.']})
    print(a.out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--evaluations',nargs='+',required=True);main(p.parse_args())
