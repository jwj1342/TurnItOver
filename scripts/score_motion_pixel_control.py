"""Offline independent rescoring of raw generations; frozen sources stay read-only."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.build_motion_grounded import read, sha, dump
from turnitover.oracle.motion_diagnosis import validate_target
from turnitover.verifier.contracts import parse_json


def score(source, prediction, frozen):
    public = json.loads(source['messages'][0]['content'].split('Public input: ', 1)[1])
    states = public['states']
    legal = {s['step'] for s in states}
    gold = source['evidence_target']
    semantic = source['condition'].startswith(('frozen_B/', 'frozen_C/'))
    original = None
    if semantic:
        original = frozen[(source['asset_id'], source['gold']['latent_class'])]
        kind = original['class']
        if source['condition'].startswith('frozen_C/') and kind in ('axis', 'clean'):
            kind = 'uncertain'
        gold = {'defect_type': kind,
                'joint': original['joint'] if kind == 'axis' else None,
                'part': original['part'] if kind in ('axis', 'offset') else None}
    error = None
    raw_kind = 'invalid'
    try:
        pred = parse_json(prediction['raw'])
        raw_kind = pred.get('defect_type', 'invalid')
        validate_target({k: v for k, v in pred.items() if k != 'comparison'})
        steps = pred['evidence_steps']
        if not all(type(s) is int for s in steps):
            raise ValueError('Non-integer evidence')
        if len(set(steps)) != len(steps) or not set(steps) <= legal:
            raise ValueError('Duplicate or nonexistent evidence')
        if 'hierarchy' in public:
            def names(node):
                return {node.get('name')} | set().union(*(names(c) for c in node.get('children', [])))
            ids = names(public['hierarchy'])
            if pred['part'] is not None and pred['part'] not in ids:
                raise ValueError('Illegal part ID')
            if pred['joint'] is not None and pred['joint'] != public.get('joint') and 'joint:'+pred['joint'] not in ids:
                raise ValueError('Illegal joint ID')
    except (ValueError, KeyError, TypeError) as exc:
        error = str(exc)
        pred = {}
        steps = []
    kind = pred.get('defect_type', 'invalid')
    correct = kind == gold['defect_type']
    joint = pred.get('joint') == gold['joint']
    part = pred.get('part') == gold['part']
    supported = False
    if correct and kind == 'uncertain':
        supported = steps == []
    elif correct and kind == 'clean':
        supported = set(steps) == legal
    elif correct and semantic:
        exposed = {s for s in legal if original['pixel_signals'][s]['exposed']}
        if kind == 'offset':
            supported = bool(set(steps) & exposed)
        elif kind == 'axis':
            supported = any(len(set(steps) & {s['step'] for s in states if s['view'] == v}) >= 2
                            and bool(set(steps) & exposed & {s['step'] for s in states if s['view'] == v})
                            for v in {s['view'] for s in states})
    elif correct:
        comparison = source['gold']['comparison']
        if kind == 'offset':
            supported = bool(steps) and set(steps) <= set(comparison['static_steps'])
        elif kind == 'axis':
            supported = any(set(p) <= set(steps) for p in comparison['motion_pairs'])
    valid = error is None
    full = valid and correct and joint and part and supported
    return {'id': source['id'], 'asset_id': source['asset_id'], 'condition': source['condition'],
            'evaluation_type': 'simulation_semantic' if semantic else 'rgb_observability_proxy',
            'expected': gold['defect_type'], 'latent': source['gold']['latent_class'],
            'predicted': kind, 'raw_predicted': raw_kind, 'raw_class_correct': raw_kind == gold['defect_type'],
            'valid': valid, 'class_correct': correct,
            'joint_correct': joint, 'part_correct': part, 'evidence_supported': supported,
            'full_correct': full, 'finding': kind in ('axis', 'offset'),
            'supported_finding': kind in ('axis', 'offset') and supported,
            'unsupported_finding': kind in ('axis', 'offset') and not full, 'error': error}


def summarize(rows):
    counts = Counter(n=len(rows))
    confusion = defaultdict(Counter)
    latent = defaultdict(Counter)
    objects = defaultdict(list)
    for r in rows:
        for k in ('valid', 'class_correct', 'raw_class_correct', 'full_correct', 'finding',
                  'supported_finding', 'unsupported_finding', 'evidence_supported'):
            counts[k] += int(r[k])
        if r['expected'] in ('axis', 'offset'):
            counts['defect_n'] += 1
            counts['defect_part_correct'] += int(r['part_correct'])
        if r['expected'] == 'axis':
            counts['axis_n'] += 1
            counts['axis_joint_correct'] += int(r['joint_correct'])
        if r['expected'] == 'uncertain':
            counts['abstention_n'] += 1
            counts['abstention_correct'] += int(r['class_correct'])
        confusion[r['expected']][r['predicted']] += 1
        latent[r['latent']][r['predicted']] += 1
        objects[r['asset_id']].append(r)
    triplets = [v for v in objects.values() if len(v) == 3]
    return {**counts, 'confusion': {k: dict(v) for k, v in confusion.items()},
            'latent_confusion': {k: dict(v) for k, v in latent.items()},
            'triplets': len(triplets),
            'collapsed_triplets': sum(len({r['predicted'] for r in v}) == 1 for v in triplets),
            'per_object': {k: {'n': len(v), 'class_correct': sum(r['class_correct'] for r in v),
                              'full_correct': sum(r['full_correct'] for r in v)} for k, v in objects.items()}}


def main(a):
    frozen = {(r['asset_id'], r['class']): r for r in read(a.frozen)}
    sources = {r['id']: r for r in read(a.dataset / 'validation.jsonl')}
    predictions = read(a.predictions)
    assert len(predictions) == len(sources) and {r['id'] for r in predictions} == set(sources)
    scored = [score(sources[p['id']], p, frozen) for p in predictions]
    groups = defaultdict(list)
    for r in scored:
        groups[r['condition']].append(r)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'scored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    dump(a.out / 'metrics.json', {'status': 'complete', 'n': len(scored),
         'dataset_sha256': sha(a.dataset/'validation.jsonl'), 'predictions_sha256': sha(a.predictions),
         'frozen_gold_sha256': sha(a.frozen), 'scorer_sha256': sha(__file__),
         'groups': {k: summarize(v) for k, v in groups.items()}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frozen', type=Path, default=Path('output/motion-attribution-v1/gold.private.jsonl'))
    main(p.parse_args())
