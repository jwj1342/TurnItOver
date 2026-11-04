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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--easy-bundle', type=Path, required=True)
    parser.add_argument('--extracted', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Intake must use a new path')
    rows = easy_intake(args.easy_bundle, args.extracted)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
