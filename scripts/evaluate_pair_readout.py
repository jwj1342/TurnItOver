"""Separate two-image visual discrimination from the full verifier output contract."""
import argparse
from collections import Counter
import json
from pathlib import Path

from turnitover.models.local_qwen import LocalQwen
from turnitover.verifier.contracts import parse_json


def run(oracle, paired, model_path, output):
    for root in (oracle,paired):
        if json.loads((root/'manifest.json').read_text())['status']!='complete':
            raise ValueError('Complete datasets required')
    pairs={r['id']:r for r in map(json.loads,(paired/'test.diagnosis.jsonl').read_text().splitlines())}
    annotations=[r for r in map(json.loads,(oracle/'annotations.private.jsonl').read_text().splitlines()) if r['split']=='test']
    output.mkdir(parents=True,exist_ok=False)
    model=LocalQwen(model_path,max_tokens=128)
    prompt='Compare the two supplied images. Do they show a visible difference in object shape, placement, or articulation? Ignore tiny antialiasing noise. Return only a JSON object with one boolean field named different. Do not diagnose a defect or choose an action.'
    report=dict(status='incomplete',scope='two_image_discrimination',model=str(model_path),
                limitations=['Strongest acquired pair is selected with privileged pixel difference.',
                             'Same-image controls are deliberately trivial negatives.',
                             'Difference labels use the configured pixel proxy, not human annotations.'])
    metrics={}
    try:
        with (output/'predictions.jsonl').open('w') as stream:
            for annotation in annotations:
                uid=annotation['id']
                trajectory=annotation['trajectories']['oracle']
                indices=trajectory['witness_indices']
                row=pairs[f'{uid}:oracle:paired']
                best=max(range(len(indices)),key=lambda i:annotation['signals']['against_reference'][indices[i]]['image_l1'])
                for condition,step in (('front',0),('strongest',best),('identical',best)):
                    reference,candidate=[paired/p for p in row['images'][2*step:2*step+2]]
                    if condition=='identical': reference=candidate
                    expected=False if condition=='identical' else annotation['signals']['against_reference'][indices[step]]['exposed']
                    response=model(prompt,[reference,candidate])
                    prediction,error=None,None
                    try:
                        value=parse_json(response.text)
                        if response.finish_reason!='completed' or set(value)!={'different'} or type(value['different']) is not bool:
                            raise ValueError('Expected a completed one-boolean JSON response')
                        prediction=value['different']
                    except ValueError as exc:
                        error=str(exc)
                    metric=metrics.setdefault(condition,Counter())
                    metric['n']+=1; metric['positive']+=int(expected)
                    if error: metric['invalid']+=1
                    elif prediction==expected: metric['correct']+=1
                    if prediction is True: metric['tp' if expected else 'fp']+=1
                    elif expected: metric['fn']+=1
                    record=dict(id=uid,condition=condition,step=step,expected=expected,prediction=prediction,error=error,raw=response.text)
                    stream.write(json.dumps(record)+'\n'); stream.flush()
                    print(json.dumps({k:v for k,v in record.items() if k!='raw'}),flush=True)
        report.update(status='complete',metrics=metrics)
    except Exception as exc:
        report.update(error=str(exc),metrics=metrics)
        raise
    finally:
        (output/'metrics.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',type=Path,default=Path('data/benchmarks/oracle-v1'))
    parser.add_argument('--paired',type=Path,default=Path('data/benchmarks/paired-v1'))
    parser.add_argument('--model',type=Path,default=Path('models/Qwen3-VL-2B-Instruct'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.oracle,args.paired,args.model,args.out),indent=2))
