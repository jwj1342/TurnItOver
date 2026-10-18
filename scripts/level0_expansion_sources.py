"""Reuse SHA-pinned, Test-disjoint structural PartNet candidates already local."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, DEFAULT_SPLITS, ROOT, sha256_path
from turnitover.assets.catalog import decode_spec
from turnitover.config import load_generate_config


def load_pool():
    cfg = load_generate_config(DEFAULT_CONFIG)
    catalog_path = Path(cfg.asset_source['manifest'])
    catalog = json.loads(catalog_path.read_text())
    splits = json.loads(DEFAULT_SPLITS.read_text())
    conversion_path = catalog_path.parent / 'conversion.json'
    conversion = json.loads(conversion_path.read_text())
    download_path = ROOT / 'data/partnet-mobility-pilot/download-manifest.json'
    download = json.loads(download_path.read_text())
    assert download['dataset_revision'] == conversion['dataset_revision']
    archives = {a['asset_id']: a for a in download['archives']}
    existing = {Path(e['spec']).stem: e for e in catalog['assets']}
    pool = []
    for row in conversion['assets']:
        aid = f"PM_{row['asset_id']}"
        group = row['source_identity']
        if row['status'] != 'structural_candidate' or row.get('frozen_test_overlap') is not False:
            continue
        if splits['groups'].get(group) == 'validation':
            continue
        path = catalog_path.parent / f'{aid}.json'
        if sha256_path(path) != row['spec_sha256']:
            raise ValueError(f'Spec SHA mismatch: {aid}')
        spec = decode_spec(json.loads(path.read_text()))
        archive = download_path.parent / archives[row['asset_id']]['path']
        if sha256_path(archive) != row['archive_sha256']:
            raise ValueError(f'Archive SHA mismatch: {aid}')
        metadata = dict(existing.get(aid, {}))
        metadata.update(split_group=group, model_id=row['model_id'],
            origin=f"{download['dataset']}@{download['dataset_revision']} {row['asset_id']}.zip",
            license=download['license'], dataset=download['dataset'], dataset_revision=download['dataset_revision'],
            source_dataset=download['dataset'], source_asset_id=row['asset_id'],
            source_identity=group, archive_sha256=row['archive_sha256'],
            original_spec_sha256=row['spec_sha256'], formal_training=False,
            original_geometry_fingerprint=row['scale_audit']['geometry_fingerprint'],
            upstream_structural_checks={'scale_audit': row['scale_audit'], 'annotation_audit': row['annotation_audit'],
                'frozen_test_overlap': False},
            previous_motion_review=row.get('motion_review'))
        pool.append({'spec': spec, 'metadata': metadata, 'original_train': group in splits['groups']})
    # Existing Train was already exhausted by v2. Expand with broad native panels first.
    simple = {'StorageFurniture', 'Door', 'Box', 'TrashCan', 'Toilet', 'Window', 'Table', 'CoffeeMachine', 'Dispenser', 'Toaster'}
    pool.sort(key=lambda r: (r['spec'].category not in simple, r['original_train'], r['spec'].asset_id))
    return pool, splits, {str(p): sha256_path(p) for p in (catalog_path, conversion_path, download_path, DEFAULT_SPLITS)}
