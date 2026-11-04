"""Adapt native Easy construction media into anonymous review intake."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def easy_intake(bundle: Path, extracted: Path) -> list[dict]:
    if extracted.exists():
        raise ValueError('Extracted mid-state directory must be new')
    extracted.mkdir(parents=True)
    result = []
    for line in (bundle / 'review_manifest.jsonl').read_text().splitlines():
        row = json.loads(line)
        cid = row['case_id']
        primary, supplementary = [], []
        for role_index, role in enumerate(('参考', '候选')):
            for index, state in enumerate(('rest', 'limit')):
                path = bundle / 'frames' / cid / f'{2 * role_index + index}.png'
                primary.append({'path': str(path.resolve()), 'caption': f'{role} / 初始视角 / {state}'})
            # Extract the unchanged 256px mid tile from the lossless native board.
            # Header/case identifiers never enter the reviewer package.
            with Image.open(bundle / row['board']) as board:
                x, y = 104 + 10 + (256 + 10), 88 + 10 + role_index * (256 + 10)
                path = extracted / f'{cid}-{role_index}-mid.png'
                board.crop((x, y, x + 256, y + 256)).save(path)
            supplementary.append({'path': str(path.resolve()), 'caption': f'{role} / 初始视角 / mid'})
        result.append({'source_case_id': cid, 'bundle': str(bundle.resolve()),
                       'primary': primary, 'supplementary': supplementary})
    return result


def hard_intake(bundle: Path, baseline_root: Path | None = None) -> list[dict]:
    baseline_ids = set()
    if baseline_root is not None:
        baseline_assets = {path.parent.name for path in
                           (baseline_root / 'private/objects').glob('*/record.json')}
        for line in (bundle / 'private/gold.jsonl').read_text().splitlines():
            gold = json.loads(line)
            if gold.get('asset_id') in baseline_assets:
                baseline_ids.add(gold['case_id'])
    result = []
    for line in (bundle / 'review_manifest.jsonl').read_text().splitlines():
        row = json.loads(line)
        cid = row['case_id']
        if not cid.startswith('hp-') or cid in baseline_ids:
            continue
        case = {'source_case_id': cid, 'bundle': str(bundle.resolve()),
                'primary': [], 'supplementary': []}
        for group, view_key in (('primary', 'initial_view'),
                                ('supplementary', 'supplementary_view')):
            view = row[view_key]
            view_label = '初始视角' if group == 'primary' else '补充视角'
            for role, filename in (('参考', 'reference'), ('候选', 'candidate')):
                for state in row['states']:
                    path = bundle / row['native_root'] / view / f'{filename}_{state}.png'
                    case[group].append({'path': str(path.resolve()),
                                        'caption': f'{role} / {view_label} / {state}'})
        result.append(case)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--easy-bundle', type=Path, action='append', default=[])
    parser.add_argument('--medium-bundle', type=Path, action='append', default=[])
    parser.add_argument('--hard-bundle', type=Path, action='append', default=[])
    parser.add_argument('--hard-baseline-root', type=Path)
    parser.add_argument('--extracted', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Intake must use a new path')
    rows = []
    for index, bundle in enumerate(args.easy_bundle + args.medium_bundle):
        if args.extracted is None:
            raise ValueError('--extracted is required for Easy and Medium bundles')
        extracted = args.extracted if len(args.easy_bundle + args.medium_bundle) == 1 else \
            args.extracted / str(index)
        rows.extend(easy_intake(bundle, extracted))
    for bundle in args.hard_bundle:
        rows.extend(hard_intake(bundle, args.hard_baseline_root))
    if not rows or len({row['source_case_id'] for row in rows}) != len(rows):
        raise ValueError('Intake must contain unique cases from at least one bundle')
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
