"""Summarize completed repairs and provider-reported costs without further API calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path


def summarize(root: Path):
    summary = json.loads((root / 'summary.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('Only completed experiments can be compared')
    groups = defaultdict(list)
    for episode in summary['episodes']:
        folder = root / episode['id'] / episode['condition']
        initial = json.loads((folder / 'initial.audit.private.json').read_text())
        result = json.loads((folder / 'result.json').read_text())
        models = [json.loads(p.read_text()) for p in sorted(folder.glob('round-*/model.json'))]
        if result['status'] != 'complete' or len(models) != result['model_calls']:
            raise ValueError(f'Incomplete response ledger: {folder}')
        for key in ('final_passed', 'first_success_round', 'model_calls', 'total_tokens'):
            if episode[key] != result[key]:
                raise ValueError(f'Summary disagrees with episode {key}: {folder}')
        costs = [m['usage'].get('cost') for m in models]
        groups[episode['condition']].append(dict(
            initial_passed=initial['passed'], **episode,
            score_trace=result['score_trace'],
            reported_cost_usd=str(sum((Decimal(str(c)) for c in costs if c is not None), Decimal(0))),
            cost_missing_calls=sum(c is None for c in costs)))
    conditions = {}
    for condition, rows in groups.items():
        damaged = [r for r in rows if not r['initial_passed']]
        clean = [r for r in rows if r['initial_passed']]
        conditions[condition] = dict(
            damaged_cases=len(damaged), repaired=sum(r['final_passed'] for r in damaged),
            clean_cases=len(clean), clean_preserved=sum(r['final_passed'] for r in clean),
            model_calls=sum(r['model_calls'] for r in rows),
            total_tokens=sum(r['total_tokens'] for r in rows),
            reported_cost_usd=str(sum((Decimal(r['reported_cost_usd']) for r in rows), Decimal(0))),
            cost_missing_calls=sum(r['cost_missing_calls'] for r in rows), episodes=rows)
    report = dict(model=summary['model'], conditions=conditions,
        reported_cost_usd=str(sum((Decimal(c['reported_cost_usd']) for c in conditions.values()), Decimal(0))),
        limitations=summary['limitations'])
    (root / 'analysis.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
