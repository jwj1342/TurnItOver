#!/usr/bin/env python3
"""Strict paired scoring of existing v4 evaluator outputs, private labels for strata only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turnitover.verifier.contracts import Context, parse_decision, parse_json


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def signature(verdict, fields):
    return Counter(tuple(tuple(sorted(getattr(finding, key))) if key in ('parts','evidence_steps') else getattr(finding, key)
                         for key in fields) for finding in verdict.findings)


def score(args):
    records = {r['id']: r for r in read(args.dataset / 'validation.episodes.jsonl') if r['terminal']}
    annotations = {r['id']: r for r in read(args.dataset / 'annotations.private.jsonl') if r['split'] == 'validation'}
    result = {'definition': 'Strict multiset row equality for each finding projection; strict_all includes status and exact evidence sets. Empty sets count correct for clean; does not imply pass verification.',
              'private_labels_usage': 'Evaluation strata only; never added to model input.', 'arms': {}, 'paired': []}
    per_arm = {}
    for arm in ('baseline','experiment'):
        predictions = read(getattr(args, arm) / 'predictions.jsonl')
        assert set(r['id'] for r in predictions) == set(records) and len(predictions) == len(records)
        buckets = {}
        per_arm[arm] = {}
        for prediction in predictions:
            record = records[prediction['id']]
            payload = json.loads(record['messages'][0]['content'].rsplit('\n',1)[-1])
            context = Context(**payload['context'])
            observed = {r['step'] for r in payload['history']}
            target = parse_decision(json.loads(record['messages'][1]['content']), context, observed, final_only=True)
            try:
                predicted = parse_decision(parse_json(prediction['raw']), context, observed, final_only=True)
            except ValueError:
                predicted = None
            checks = {'verdict': bool(predicted and predicted.status == target.status)}
            projections = {'defect_id': ('defect_id',), 'joint_id': ('defect_id','joint_id'),
                           'parts': ('defect_id','joint_id','parts'), 'evidence': ('defect_id','joint_id','parts','evidence_steps')}
            checks.update({key: bool(predicted and signature(predicted, fields) == signature(target, fields)) for key,fields in projections.items()})
            checks['strict_all'] = checks['verdict'] and checks['evidence']
            checks['exact_verdict'] = checks['verdict'] and checks['parts']
            checks['evidence_supported'] = prediction['evidence_consistent'] and checks['exact_verdict']
            labels = sorted({r['defect_id'] for r in annotations[record['episode_id']]['labels']})
            category = '+'.join(labels) if labels else 'clean'
            item = {'id': record['id'], 'asset_id': record['data_identity']['asset_id'], 'category': category, **checks}
            per_arm[arm][record['id']] = item
            for bucket in ('all',category):
                count = buckets.setdefault(bucket, Counter())
                count['n'] += 1
                count.update({key:int(value) for key,value in checks.items()})
        result['arms'][arm] = {key:dict(value) for key,value in buckets.items()}
    for uid in records:
        before, after = per_arm['baseline'][uid], per_arm['experiment'][uid]
        result['paired'].append({'id':uid, 'asset_id':before['asset_id'], 'category':before['category'],
                                'baseline':before, 'experiment':after,
                                'strict_all_change':int(after['strict_all'])-int(before['strict_all']),
                                'exact_verdict_change':int(after['exact_verdict'])-int(before['exact_verdict'])})
    result['validation_assets'] = sorted({r['data_identity']['asset_id'] for r in records.values()})
    result['validation_episodes_sha256'] = hashlib.sha256((args.dataset/'validation.episodes.jsonl').read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='paired'}, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','baseline','experiment','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    score(parser.parse_args())
