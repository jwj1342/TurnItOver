"""Freeze a train-only pixel threshold before any validation repair results."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from batch_common import MANIFEST,dump,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    assert sha(a.dataset/'manifest.json')==MANIFEST
    source=a.dataset/'train.episodes.jsonl'
    rows=[]
    for r in map(json.loads,source.read_text().splitlines()):
        groups={}
        for name in r['images']:
            tail=Path(name).name.split('-',1)[1]
            role='reference' if 'reference_' in tail else 'candidate'
            key=tail.replace(role+'_','',1)
            groups.setdefault(key,{})[role]=name
        scores=[]
        for pair in groups.values():
            assert set(pair)=={'reference','candidate'}
            ref=np.asarray(Image.open(a.dataset/pair['reference']).convert('RGB'),dtype=np.float32)
            cand=np.asarray(Image.open(a.dataset/pair['candidate']).convert('RGB'),dtype=np.float32)
            scores.append(float(np.abs(ref-cand).mean()/255.))
        verdict=json.loads(r['messages'][-1]['content'])['verdict']['status']
        rows.append({'case_id':r['episode_id'],'target':verdict,'score':max(scores),'pair_count':len(scores)})
    fitted=[r for r in rows if r['target'] in ('pass','fail')]
    scores=np.array([r['score'] for r in fitted]);y=np.array([r['target']=='fail' for r in fitted])
    values=np.unique(scores)
    thresholds=np.unique(np.r_[0.,(values[:-1]+values[1:])/2,1.])
    results=[]
    for t in thresholds:
        pred=scores>t
        balanced=float(((pred[y]).mean()+(~pred[~y]).mean())/2)
        results.append((balanced,float(t)))
    best=max(x[0] for x in results);threshold=min(t for s,t in results if s==best)
    assert len(rows)==600 and len(fitted)==579 and int(y.sum())==279
    dump(a.output,{'status':'frozen','split':'train','metric':'mean_absolute_rgb_0_1','aggregate':'max over all frozen reference/candidate pairs',
        'threshold':threshold,'training_manifest_sha256':MANIFEST,'training_split_sha256':sha(source),
        'selection':'maximize train balanced accuracy; smallest threshold breaks ties; fail iff score > threshold',
        'labels':'frozen Evidence pass/fail; 21 uncertain excluded from threshold fitting',
        'n_train':600,'n_fit':579,'pass':300,'fail':279,'uncertain_excluded':21,
        'train_balanced_accuracy':best,'records':rows})
    print('PIXEL_CALIBRATED',threshold,best,flush=True)
if __name__=='__main__':main()
