"""Minimal defect-ID readout on static and strongest acquired states, without verdict prose."""
import argparse
from collections import Counter
import json
from pathlib import Path

from turnitover.models.local_qwen import LocalQwen
from turnitover.verifier.contracts import parse_json

IDS=('structure.part_offset','kinematics.joint_axis')


def run(oracle,paired,model_path,output):
    for root in (oracle,paired):
        if json.loads((root/'manifest.json').read_text())['status']!='complete': raise ValueError('Incomplete dataset')
    pairs={r['id']:r for r in map(json.loads,(paired/'test.diagnosis.jsonl').read_text().splitlines())}
    annotations=[r for r in map(json.loads,(oracle/'annotations.private.jsonl').read_text().splitlines()) if r['split']=='test']
    output.mkdir(parents=True,exist_ok=False)
    model=LocalQwen(model_path,max_tokens=256)
    report=dict(status='incomplete',scope='minimal_defect_classification',model=str(model_path),
                limitations=['Strongest acquired state is selected using privileged pixel difference.',
                             'No localization, severity, stopping or repair prediction.',
                             'Unexposed full-gold labels still count as misses.'])
    metrics={}
    try:
        with (output/'predictions.jsonl').open('w') as stream:
            for annotation in annotations:
                uid=annotation['id']; trajectory=annotation['trajectories']['oracle']; indices=trajectory['witness_indices']
                best=max(range(len(indices)),key=lambda i:annotation['signals']['against_reference'][indices[i]]['image_l1'])
                row=pairs[f'{uid}:oracle:paired']
                front_ref,front_cand=[paired/p for p in row['images'][:2]]
                best_ref,best_cand=[paired/p for p in row['images'][2*best:2*best+2]]
                state=annotation['grid'][indices[best]]
                for condition in ('single','paired'):
                    paths=[front_ref,front_cand,best_cand] if condition=='single' else [front_ref,front_cand,best_ref,best_cand]
                    order='reference at rest, candidate at rest, candidate at acquired state' if condition=='single' else 'reference at rest, candidate at rest, reference at acquired state, candidate at acquired state'
                    prompt=f'Inspect this articulated object. The images are ordered as: {order}. The acquired state is {json.dumps(state)}. Classify only visible errors from these images. Allowed defect IDs: structure.part_offset means a part is misplaced relative to its parent; kinematics.joint_axis means articulation follows the wrong axis. Do not infer an error solely because motion was requested. Return only JSON with one field defect_ids, a list of supported IDs, empty if none is established. Do not output actions, explanations, confidence, or a verdict.'
                    response=model(prompt,paths)
                    predicted=set(); error=None
                    try:
                        value=parse_json(response.text)
                        if response.finish_reason!='completed' or set(value)!={'defect_ids'} or not isinstance(value['defect_ids'],list) or any(type(x) is not str or x not in IDS for x in value['defect_ids']):
                            raise ValueError('Expected completed defect_ids list')
                        predicted=set(value['defect_ids'])
                    except ValueError as exc: error=str(exc)
                    expected={c['defect_id'] for c in annotation['labels']}
                    exposed={annotation['labels'][i]['defect_id'] for index in (indices[0],indices[best]) for i in annotation['exposures'][index]}
                    metric=metrics.setdefault(condition,dict(n=0,invalid=0,classes={label:Counter() for label in IDS}))
                    metric['n']+=1; metric['invalid']+=int(error is not None)
                    for label in expected|predicted:
                        metric['classes'][label]['tp' if label in expected&predicted else 'fp' if label in predicted else 'fn']+=1
                    record=dict(id=uid,condition=condition,expected=sorted(expected),exposed=sorted(exposed),predicted=sorted(predicted),error=error,raw=response.text)
                    stream.write(json.dumps(record)+'\n');stream.flush()
                    print(json.dumps({k:v for k,v in record.items() if k!='raw'}),flush=True)
        report.update(status='complete',metrics=metrics)
    except Exception as exc:
        report.update(error=str(exc),metrics=metrics)
        raise
    finally:(output/'metrics.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',type=Path,default=Path('data/benchmarks/oracle-v1'))
    parser.add_argument('--paired',type=Path,default=Path('data/benchmarks/paired-v1'))
    parser.add_argument('--model',type=Path,default=Path('models/Qwen3-VL-2B-Instruct'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.oracle,args.paired,args.model,args.out),indent=2))
