"""Private, post-inference attribution scoring; keep class and protocol scores separate."""
import argparse
from collections import Counter
import json
from pathlib import Path
from scripts.evaluate_modular_verifier import _read_jsonl
from turnitover.verifier.contracts import parse_json
from turnitover.verifier.modular import parse_verdict

CLASSES={'kinematics.joint_axis':'axis','structure.part_offset':'offset'}


def normalize(raw, condition, record):
    value=parse_json(raw)
    if condition=='A':
        payload=json.loads(record['messages'][0]['content'].rsplit('\n',1)[1])
        parse_verdict(raw,payload['context'],payload['history'])
        v=value['verdict'];fs=v['findings'];f=fs[0] if len(fs)==1 else {}
        return {'abnormal':{'fail':True,'pass':False,'uncertain':None}[v['status']],
                'classification':CLASSES.get(f.get('defect_id'),'clean' if v['status']=='pass' else 'uncertain'),
                'joint_id':f.get('joint_id'),'part_id':(f.get('parts') or [None])[0],
                'evidence_steps':f.get('evidence_steps',[]),'explanation':v['summary']},True
    keys={'abnormal','classification','joint_id','part_id','evidence_steps','explanation'}
    valid=set(value)==keys and value.get('classification') in ('axis','offset','clean','uncertain')
    valid=valid and (type(value.get('abnormal')) is bool or value.get('abnormal') is None)
    valid=valid and isinstance(value.get('evidence_steps'),list)
    if valid:
        allowed={0,3} if condition=='C' else set(range(6))
        valid=all(type(i) is int and i in allowed for i in value['evidence_steps'])
    if value.get('classification') in ('axis','offset'):
        valid=valid and isinstance(value.get('part_id'),str) and bool(value.get('evidence_steps'))
        valid=valid and (isinstance(value.get('joint_id'),str) if value['classification']=='axis' else value.get('joint_id') is None)
    if value.get('classification') in ('clean','uncertain'):
        valid=valid and value.get('joint_id') is None and value.get('part_id') is None
    return value,valid


def score(pred,gold,record):
    condition=pred['condition'];kind=gold['class']
    expected='uncertain' if condition=='C' and kind in ('axis','clean') else kind
    try: value,valid=normalize(pred['raw'],condition,record);error=None
    except (ValueError,KeyError,TypeError) as exc: value={};valid=False;error=str(exc)
    cls=value.get('classification');steps=value.get('evidence_steps',[])
    allowed={0,3} if condition=='C' else set(range(6))
    real=isinstance(steps,list) and bool(steps) and all(type(i)==int and i in allowed for i in steps)
    kind_ok=cls==expected
    joint_ok=value.get('joint_id')==gold['joint']
    part_ok=value.get('part_id')==gold['part']
    supporting=False
    if real and kind_ok and cls in ('axis','offset'):
        if cls=='offset': supporting=any(gold['pixel_signals'][s]['exposed'] for s in steps)
        elif condition in ('B','D'):
            # Multiple different values within a fixed camera, with visible mismatch.
            supporting=any(len(set(steps)&set(group))>=2 and
                any(gold['pixel_signals'][s]['exposed'] for s in set(steps)&set(group))
                for group in ((0,1,2),(3,4,5)))
    full=valid and kind_ok and joint_ok and part_ok
    if cls in ('axis','offset'):full=full and supporting
    expected_abnormal=None if expected=='uncertain' else expected!='clean'
    return {'id':pred['id'],'asset_id':pred['asset_id'],'condition':condition,'gold':kind,
            'expected_visible':expected,'predicted':cls,'protocol_valid':bool(valid),'parse_error':error,
            'abnormal_correct':value.get('abnormal')==expected_abnormal and bool(value),
            'class_correct':kind_ok,'joint_correct_axis':joint_ok if kind=='axis' else None,
            'part_correct_defect':part_ok if kind!='clean' else None,'full_correct':bool(full),
            'citations_real':real,'evidence_supports_type':supporting,
            'evidence_supports_full_finding':supporting and joint_ok and part_ok,
            'unsupported_finding':cls in ('axis','offset') and not (supporting and joint_ok and part_ok),
            'uncertain':cls=='uncertain','unsupported_clean':condition=='C' and cls=='clean',
            'truncated':pred['truncated']}


def summary(rows):
    keys=('protocol_valid','abnormal_correct','class_correct','full_correct','citations_real',
          'evidence_supports_type','evidence_supports_full_finding','unsupported_finding','uncertain','unsupported_clean')
    return {'n':len(rows),**{k:sum(bool(r[k]) for r in rows) for k in keys},
       'joint_axis':{'correct':sum(r['joint_correct_axis'] is True for r in rows),'n':sum(r['joint_correct_axis'] is not None for r in rows)},
       'part_defect':{'correct':sum(r['part_correct_defect'] is True for r in rows),'n':sum(r['part_correct_defect'] is not None for r in rows)},
       'confusion':dict(Counter(r['gold']+'->'+str(r['predicted']) for r in rows))}


def main(root):
    gold={r['id']:r for r in _read_jsonl(root/'gold.private.jsonl')}
    inputs={(r['id'],r['condition']):r for name in ('inputs.jsonl','controls.inputs.jsonl') for r in _read_jsonl(root/name)}
    report={};all_scores=[]
    for model in ('base','v4'):
        rows=[]
        for folder in (model,model+'-D'):
            for p in _read_jsonl(root/folder/'predictions.jsonl'):
                r=score(p,gold[p['id']],inputs[p['id'],p['condition']]);r['model']=model;rows.append(r)
        all_scores.extend(rows)
        report[model]={c:summary([r for r in rows if r['condition']==c]) for c in ('A','B','C','D')}
        report[model]['per_object_B']={a:summary([r for r in rows if r['condition']=='B' and r['asset_id']==a]) for a in sorted({r['asset_id'] for r in rows})}
        report[model]['B_triplets_all_correct']=sum(all(r['class_correct'] for r in rows if r['condition']=='B' and r['asset_id']==a) for a in sorted({r['asset_id'] for r in rows}))
    (root/'scores.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_scores))
    (root/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
