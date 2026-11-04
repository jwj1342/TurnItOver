"""Build private Easy candidates with the unchanged native-rendering gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.tier_construction import construct_easy_batch as builder
from scripts.tier_construction import level0_expansion_sources as sources


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, action='append', required=True)
    parser.add_argument('--download-manifest', type=Path, action='append', required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--exclude-gold', type=Path, action='append', required=True)
    parser.add_argument('--exclude-source-list', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--specs', type=Path, required=True)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--max-tries', type=int, default=1000)
    args = parser.parse_args()
    if len(args.catalog) != len(args.download_manifest) or args.target < 1:
        parser.error('Match every catalog to a download manifest; target must be positive')
    out, specs = args.out.resolve(), args.specs.resolve()
    if out == specs or out.is_relative_to(specs) or specs.is_relative_to(out):
        parser.error('Output and specs must be separate directories')
    excluded = set()
    for path in args.exclude_gold:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            source = row.get('source_identity') or row.get('source', {}).get('source_identity')
            asset = row.get('asset_id')
            if source:
                excluded.add('PM_' + source.rsplit(':', 1)[-1])
            elif asset:
                excluded.add(asset)
            else:
                raise ValueError(f'Gold row has no source identity: {path}')
    for path in args.exclude_source_list:
        excluded.update('PM_' + value.rsplit(':', 1)[-1] for value in json.loads(path.read_text()))
    inputs = [Path(__file__), *Path(__file__).with_name('tier_construction').glob('*.py'),
              *args.catalog, *args.download_manifest, args.splits, args.config,
              *args.exclude_gold, *args.exclude_source_list]
    for catalog in args.catalog:
        inputs.append(catalog.parent / 'conversion.json')
    pins = {str(path.resolve()): sha(path) for path in inputs}
    state = {'version': 1, 'target': args.target, 'max_tries': args.max_tries,
             'inputs': pins, 'excluded_assets': sorted(excluded), 'specs': str(specs),
             'policy': 'inherited-easy-native256-pending-human'}
    state_path = out / 'construction-inputs.json'
    if out.exists() and any(out.iterdir()):
        if not state_path.exists() or json.loads(state_path.read_text()) != state:
            raise ValueError('Nonempty output has different or unpinned construction inputs')
    out.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    sources.DEFAULT_SPLITS = args.splits.resolve()
    builder.DEFAULT_CONFIG = args.config.resolve()
    builder.SOURCES = tuple(zip(args.catalog, args.download_manifest))
    builder.prior_objects = lambda: excluded
    report = builder.build(out, specs, args.target, args.max_tries)
    for path, digest in pins.items():
        if sha(Path(path)) != digest:
            raise ValueError('Construction input changed: ' + path)
    print(json.dumps(report, indent=2, sort_keys=True))
    # Inherited summary names differ between historical builders; use actual rows.
    accepted = sum(row['status'] == 'automatic_pass' for row in builder.rows(out / 'private/screening.jsonl'))
    raise SystemExit(0 if accepted >= args.target else 2)


if __name__ == '__main__':
    main()
