#!/usr/bin/env python3
"""Offline accounting of completed pairs and every successful/failed API attempt."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic
from turnitover.repair.loop import usage_tokens


def summarize(runs, ledgers):
    if len({p.resolve() for p in ledgers}) != len(ledgers):
        raise ValueError('Duplicate ledger would double count costs')
    calls = []
    for path in ledgers:
        ledger = json.loads(path.read_text())
        if ledger['request_count'] != len(ledger['calls']):
            raise ValueError('Ledger count does not match attempts')
        for item in ledger['calls']:
            calls.append({'ledger': str(path), **item})
    costs = [c.get('provider_cost') for c in calls]
    summary = {'api_accounting': {
        'request_attempts': len(calls), 'request_status': dict(Counter(c['status'] for c in calls)),
        'reported_tokens': sum(usage_tokens(c.get('usage', {})) for c in calls),
        'attempts_without_usage': sum(not bool(c.get('usage')) for c in calls),
        'reported_cost_count': sum(c is not None for c in costs),
        'provider_cost_total': sum(costs) if calls and all(type(c) in (int, float) for c in costs) else None,
        'elapsed_ms': sum(c.get('elapsed_ms', 0) for c in calls)},
        'calls': calls, 'runs': [], 'paired_rows': [],
        'limitations': ['Missing usage/cost is unknown, never zero.',
                        'Geometric witness consistency does not establish full semantic evidence correctness.']}
    for run in runs:
        result = json.loads((run / 'result.json').read_text())
        summary['runs'].append({'path': str(run), 'status': result['status'], 'error': result.get('error')})
        score_path = run / 'scores.private.json'
        if result['status'] != 'complete' or not score_path.exists():
            continue
        scored = json.loads(score_path.read_text())
        for case_id, arms in scored['pairs'].items():
            for mode, arm in arms.items():
                summary['paired_rows'].append({'run': str(run), 'case': case_id, 'mode': mode,
                    'initial_inputs_equal': result['pairs'][case_id]['initial_inputs_equal'],
                    'initial_labels_valid': arm['rounds'][0]['labels_valid'],
                    'diagnosis': arm['rounds'][0].get('diagnosis'),
                    'final_code_correct': arm['final_quality']['code_correct'],
                    'first_correct_round': arm['first_correct_round'],
                    'verifier_status': arm['status'], 'verifier_accepted': arm['verifier_accepted'],
                    'visual_repair_calls': arm['visual_repair_calls'],
                    'total_observations': arm['total_observations'], 'api_cost': arm['api_cost'],
                    'actions': [r['actions'] for r in arm['rounds']]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, default=[])
    parser.add_argument('--ledger', action='append', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output must be new')
    write_json_atomic(args.out, summarize(args.run, args.ledger))
    print(args.out)


if __name__ == '__main__':
    main()
