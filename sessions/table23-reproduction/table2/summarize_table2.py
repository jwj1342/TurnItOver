"""Audit all 21 Table 2 runs against frozen targets; aggregate three seeds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
from table2_projection import project

HASH = 'b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b'
SEEDS = (20261013, 20261113, 20261213)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def recount(rows, labels):
    scores = {}
    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            target = any(f[0] == label for f in row['expected_findings'])
            pred = any(f[0] == label for f in row['predicted_findings'])
            tp += target and pred
            fp += pred and not target
            fn += target and not pred
        scores[label] = {'tp': tp, 'fp': fp, 'fn': fn,
                         'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    return mean(s['f1'] for s in scores.values()), scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'runs', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Output file must not already exist')
    manifest = read(a.dataset / 'manifest.json')
    assert hashlib.sha256((a.dataset / 'manifest.json').read_bytes()).hexdigest() == HASH
    path = a.dataset / 'validation.episodes.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files'][path.name]
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    records = [r for r in records if r['admission'] == 'eligible']
    assert len(records) == 104 and Counter(r['is_clean'] for r in records) == {True: 60, False: 44}
    labels = sorted(manifest['experiment_taxonomy'])
    assert len(labels) == 5
    result = {'manifest_sha256': HASH, 'units': 'fraction', 'interval': 'training seed min/max, not confidence interval', 'rows': []}
    audited = 0
    for condition in ('full', 'motion', 'viewpoint'):
        projected = [(r, r['images']) if condition == 'full' else project(r, r['images'], condition) for r in records]
        targets = {pair[0]['id']: pair[0] for pair in projected if pair is not None}
        assert len(targets) == (30 if condition == 'viewpoint' else 104)
        input_digests = set()
        for mode in ('base', 'ordinary', 'evidence'):
            tier_values = {tier: [] for tier in ('all', 'easy', 'medium', 'hard')}
            keys = ['base'] if mode == 'base' else [f'{mode}-{seed}' for seed in SEEDS]
            for key in keys:
                folder = a.runs / f'{condition}-{key}'
                metrics = read(folder / ('metrics.json' if condition == 'full' else 'table2-f1.json'))
                rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
                assert metrics['status'] == 'complete' and metrics['n'] == len(rows) == len(targets)
                assert len({r['id'] for r in rows}) == len(rows) and {r['id'] for r in rows} == set(targets)
                assert metrics['protocol']['dataset_manifest_sha256'] == HASH
                input_digests.add(metrics['protocol']['public_input_sha256'])
                for row in rows:
                    record = targets[row['id']]
                    findings = json.loads(record['messages'][1]['content'])['verdict']['findings']
                    expected = {(f['defect_id'], f.get('joint_id'), tuple(sorted(f['parts']))) for f in findings}
                    actual = {(f[0], f[1], tuple(sorted(f[2]))) for f in row['expected_findings']}
                    assert expected == actual and row['clean'] == record['is_clean']
                    assert row['difficulty'] == record['difficulty']
                for tier in tier_values:
                    subset = rows if tier == 'all' else [r for r in rows if r['difficulty'] == tier]
                    if not subset:
                        continue
                    reported = metrics if tier == 'all' else metrics['by_difficulty'][tier]
                    f1, per_class = recount(subset, labels)
                    assert abs(f1 - reported['defect_macro_f1']) < 1e-12
                    for label in labels:
                        for field in ('tp', 'fp', 'fn'):
                            assert per_class[label][field] == reported['defect_per_class'][label][field]
                    tier_values[tier].append(f1)
                audited += len(rows)
            result['rows'].append({'condition': condition, 'model': mode, 'n': len(targets),
                'f1': {tier: {'mean': mean(values), 'min': min(values), 'max': max(values)} if values else None
                       for tier, values in tier_values.items()}})
        assert len(input_digests) == 1, 'Models did not receive identical public inputs'
    result.update(status='complete', audited_runs=21, audited_predictions=audited)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: 21 runs, {audited} predictions, frozen targets and five-label F1')


if __name__ == '__main__':
    main()
