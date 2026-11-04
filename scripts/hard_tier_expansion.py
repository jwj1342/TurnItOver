"""Extend a SHA-sealed full-motion Hard bundle; all output awaits human review.

Uses the separately pinned construction checkout without modifying it. Source
records, source bytes, construction modules and the pool are inventoried in the
private run state. Existing defect bytes and source split membership are kept.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from hard_tier_expansion_artifacts import checkpoint, pair_control, read, sha, write


def run(args):
    builder_root, base, pool_path, out = [p.resolve() for p in
        (args.builder_root, args.base, args.pool, args.out)]
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out)
           for p in (builder_root, base, pool_path.parent)):
        raise ValueError('Output must be isolated from all source paths')
    sys.path.insert(0, str(builder_root))
    from scripts import collect_hard_formal as builder
    from turnitover.assets.catalog import decode_spec
    from turnitover.core.program import ObjectProgram
    from turnitover.config import load_generate_config
    from turnitover.render.session import ObservationSession
    from turnitover.render.views import load_views
    if args.target < 150 or args.target % 3:
        raise ValueError('Hard target must be >=150 and divisible by three')
    pool = read(pool_path)
    by_id = {r['asset_id']: r for r in pool}
    if len(pool) != len(by_id):
        raise ValueError('Duplicate source in pool')
    baseline = read(base / 'manifest.json')
    if baseline['policy'] != builder.POLICY or baseline['objects'] != 75:
        raise ValueError('Expected the immutable 75-object full-motion baseline')
    pins = {str(base / 'manifest.json'): sha(base / 'manifest.json'), str(pool_path): sha(pool_path)}
    acquisition_manifests = {}
    official_verification_path = Path(__file__).resolve().parents[1] / 'output/hard-official-archive-verification-v1.json'
    official_verification = read(official_verification_path) if official_verification_path.exists() else None
    if official_verification:
        pins[str(official_verification_path.resolve())] = sha(official_verification_path)
        for verified in official_verification['archives']:
            acquisition_manifests[str(verified['asset_id'])] = ('official_verification', verified)
    for root in (builder_root / 'output').glob('hard-new-download-*'):
        manifest_path = root / 'download-manifest.json'
        if manifest_path.exists():
            manifest = read(manifest_path)
            if manifest.get('status') == 'complete':
                entries = {str(row['asset_id']): row for row in manifest['archives']}
                for record_path, digest in manifest.get('records', {}).items():
                    record_path = Path(record_path)
                    if sha(record_path) != digest:
                        raise ValueError('Acquisition qualified record changed: ' + str(record_path))
                    record = read(record_path)
                    aid = str(record.get('metadata', {}).get('source_asset_id'))
                    if aid in entries:
                        acquisition_manifests[aid] = (manifest_path, entries[aid])
    for name, digest in baseline['files'].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base) or sha(path) != digest:
            raise ValueError('Baseline artifact changed: ' + name)
    for path in [*builder_root.glob('scripts/*.py'), *builder_root.glob('turnitover/**/*.py'),
                 Path(__file__), Path(__file__).with_name('hard_tier_expansion_artifacts.py')]:
        pins[str(path.resolve())] = sha(path)
    exclusions = set()
    for path in args.exclude:
        document = read(path)
        exclusions.update(document.get('source_identities', []))
        pins[str(path.resolve())] = sha(path)
    atomic_pool = []
    for row in pool:
        source = read(row['source_record'])
        pins[str(Path(row['source_record']).resolve())] = sha(row['source_record'])
        if 'metadata' not in source:
            continue
        atomic_pool.append(row)
        metadata = source['metadata']
        if metadata['source_identity'] != row['source_identity']:
            raise ValueError('Source identity mismatch')
        for path, expected in source['files'].items():
            if sha(path) != expected:
                raise ValueError('Source file changed: ' + path)
            pins[str(Path(path).resolve())] = expected
        for field, key in [('source_spec', 'original_spec_sha256'), ('source_archive', 'archive_sha256')]:
            path = Path(metadata[field]).resolve()
            if sha(path) != metadata[key]:
                raise ValueError('Source provenance changed: ' + str(path))
            pins[str(path)] = metadata[key]
        archive_manifest, archive_entry = acquisition_manifests.get(str(metadata['source_asset_id']), (None, None))
        if archive_manifest == 'official_verification':
            manifest_ok = (archive_entry['sha256'] == metadata['archive_sha256']
                and Path(archive_entry['path']).resolve() == Path(metadata['source_archive']).resolve()
                and Path(archive_entry['path']).stat().st_size == archive_entry['bytes']
                and archive_entry['status'] == 'verified_against_official_lfs')
        else:
            manifest_ok = (archive_entry is not None and archive_entry['sha256'] == metadata['archive_sha256']
                and Path(archive_entry['path']).resolve() == Path(metadata['source_archive']).resolve()
                and Path(archive_entry['path']).stat().st_size == archive_entry['bytes'])
        if not manifest_ok:
            raise ValueError('Source archive lacks an acquisition-manifest binding: ' + row['asset_id'])
        if archive_manifest != 'official_verification':
            pins[str(archive_manifest.resolve())] = sha(archive_manifest)
        pins[str(Path(row['source_record']).resolve())] = sha(row['source_record'])
    pool = [r for r in atomic_pool if r['source_identity'] not in exclusions]
    state = {'version': 1, 'policy': builder.POLICY, 'target': args.target,
             'baseline_objects': 75, 'inputs': pins,
             'difficulty_status': 'pending_human_review',
             'category_caps': {k: v * args.target // 75 for k, v in builder.CATEGORY_CAPS.items()}}
    state_path = out / 'private/run_state.json'
    if state_path.exists() and read(state_path) != state:
        raise ValueError('Inputs changed; use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    write(state_path, state)
    # Diversity caps scale with the object denominator; difficulty gates stay fixed.
    builder.CATEGORY_CAPS = state['category_caps']
    default_cap = 5 * args.target // 75
    builder.category_cap = lambda category: builder.CATEGORY_CAPS.get(category, default_cap)
    def program(document):
        return ObjectProgram.from_spec(decode_spec(document))
    for original in sorted(base.glob('private/objects/*/record.json')):
        record = read(original)
        if record['asset_id'] not in by_id:
            raise ValueError('Baseline source missing from source pool')
        target_path = out / original.relative_to(base)
        if target_path.exists():
            continue
        for name, digest in record['files'].items():
            source, destination = base / name, out / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha(destination) != digest:
                raise ValueError('Baseline copy mismatch')
        pair_control(out, record, by_id[record['asset_id']]['source_spec'], program)
        write(target_path, record)
    print(json.dumps({'event': 'baseline_paired', **checkpoint(out, state, args.target)}), flush=True)
    cfg = load_generate_config(builder.DEFAULT_CONFIG)
    if (cfg.render.width, cfg.render.height) != (256, 256):
        raise ValueError('Native 256 rendering is required')
    views_tuple = load_views(cfg.views_path)
    views = {str(v.id): v for v in views_tuple}
    with ObservationSession(cfg.render, views_tuple) as session:
        for _ in range(args.limit):
            records = builder.load_records(out)
            accepted = [r for r in records if r['status'] == 'automatic_pass']
            if len(accepted) >= args.target:
                break
            choice = builder.choose(pool, records, args.target)
            if choice is None:
                break
            kind, row, target = choice
            print(json.dumps({'event': 'start', 'asset_id': row['asset_id'], 'kind': kind}), flush=True)
            record = builder.collect_one(session, views_tuple, views, out, row, target, kind)
            if record['status'] == 'automatic_pass':
                index = len(accepted) - 75
                split = 'train' if index < 25 else 'validation' if index < 45 else 'test'
                record['gold']['source']['split'] = split
                pair_control(out, record, row['source_spec'], program)
                write(out / f'private/objects/{row["asset_id"]}/record.json', record)
            report = checkpoint(out, state, args.target)
            print(json.dumps({'event': 'result', 'asset_id': row['asset_id'],
                              'result': record['status'], **report}), flush=True)
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Input changed during collection: ' + path)
    return checkpoint(out, state, args.target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builder-root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exclude', type=Path, action='append', default=[])
    parser.add_argument('--target', type=int, default=150)
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['objects'] >= args.target else 2)
