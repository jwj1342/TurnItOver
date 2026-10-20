"""Audit local Level 0 expansion capacity without rendering or model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.convert_partnet_mobility import _test_identities
from scripts.level0_expansion_sources import load_pool
from scripts.prepare_obvious_defect_review import ROOT, sha256_path

BASE = ROOT / 'docs/evidence/partnet-mobility-level0-axis-v2'


def audit(base: Path, out: Path) -> dict:
    """Emit a conservative untried queue; an upper bound never implies quality."""
    pool, splits, provenance = load_pool()  # Verifies source archives and specs.
    source = ROOT / 'data/assets/partnet-mobility-trainval-v1'
    conversion = json.loads((source / 'conversion.json').read_text())
    download_path = ROOT / 'data/partnet-mobility-pilot/download-manifest.json'
    download = json.loads(download_path.read_text())
    structural = [r for r in conversion['assets'] if r['status'] == 'structural_candidate']
    manifest = json.loads((base / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha256_path(base / name) != digest:
            raise ValueError(f'Base SHA mismatch: {name}')
    catalog = json.loads((base / 'private/source_catalog/catalog.json').read_text())
    used = {r['source_asset_id'] for r in catalog['assets']}
    previous, historical = set(), {}
    for name in ('screening.json', 'inherited_screening.json', 'expansion_screening.json'):
        path = base / 'private' / name
        rows = json.loads(path.read_text())['assets']
        previous.update(r['asset_id'].removeprefix('PM_') for r in rows)
        historical[name] = dict(Counter(r.get('status', 'inherited') for r in rows))
        provenance[str(path)] = sha256_path(path)
    test_plan = ROOT / 'configs/uniphys-test.json'
    test_root = ROOT / 'data/assets/uniphys/data'
    if conversion['formal_test_plan_sha256'] != sha256_path(test_plan):
        raise ValueError('Frozen Test plan differs from conversion provenance')
    test = _test_identities(test_plan, test_root)
    test_record = json.loads(test_plan.read_text())
    test_content, test_models = set(), set()
    for row in test_record['objects']:
        folder = test_root / row['id']
        metadata = json.loads((folder / 'meta_data.json').read_text())
        test_models.add(str(metadata['source']['original_id']))
        geometry_sha = sha256_path(folder / 'full_model/model.obj')
        if geometry_sha != row['geometry_sha256']:
            raise ValueError(f"Frozen Test geometry SHA mismatch: {row['id']}")
        test_content.add(geometry_sha)
        for key in ('geometry_sha256', 'source_identity_sha256', 'source_content_sha256', 'archive_sha256'):
            if row.get(key):
                test_content.add(row[key])
    protected = [r for r in structural if r['asset_id'] in used or r['source_identity'] in splits['groups']]
    keys = ('source_identity', 'model_id', 'archive_sha256', 'spec_sha256')
    protected_values = {k: {r[k] for r in protected} for k in keys}
    protected_geometry = {r['scale_audit']['geometry_fingerprint'] for r in protected}
    seen = {k: set() for k in keys}
    seen_geometry = set()
    queue, excluded = [], []
    for candidate in pool:
        meta = candidate['metadata']
        aid = meta['source_asset_id']
        row = next(r for r in structural if r['asset_id'] == aid)
        reasons = []
        if aid in used:
            reasons.append('already_in_v2')
        elif aid in previous:
            reasons.append('historically_screened_not_fresh_capacity')
        if (row.get('motion_review') or {}).get('decision') == 'failed':
            reasons.append('previous_normal_motion_review_failed')
        fp = row['scale_audit']['geometry_fingerprint']
        for key in keys:
            if aid not in used and row[key] in protected_values[key]:
                reasons.append('protected_train_validation_duplicate_' + key)
            if row[key] in seen[key]:
                reasons.append('pool_duplicate_' + key)
        if aid not in used and fp in protected_geometry:
            reasons.append('protected_train_validation_duplicate_geometry')
        if fp in seen_geometry:
            reasons.append('pool_duplicate_geometry')
        if row['source_identity'] in test['source_ids'] or row['model_id'] in test_models:
            reasons.append('frozen_test_identity_overlap')
        if fp in test['geometry']:
            reasons.append('frozen_test_geometry_overlap')
        if row['archive_sha256'] in test_content or row['spec_sha256'] in test_content:
            reasons.append('frozen_test_content_overlap')
        record = {k: row[k] for k in ('asset_id', 'source_identity', 'model_id', 'category', 'archive_sha256', 'spec_sha256')}
        record.update(geometry_fingerprint=fp, previous_motion_decision=(row.get('motion_review') or {}).get('decision'),
                      source_split=splits['groups'].get(row['source_identity'], 'unassigned'))
        if reasons:
            excluded.append({**record, 'reasons': reasons})
        else:
            queue.append({**record, 'status': 'pending_normal_motion_and_level0_screening', 'candidate_split': 'train'})
            for key in keys:
                seen[key].add(row[key])
            seen_geometry.add(fp)
    validation = [r for r in structural if splits['groups'].get(r['source_identity']) == 'validation']
    archives = {row['asset_id']: row for row in download['archives']}
    for row in validation:
        archive = download_path.parent / archives[row['asset_id']]['path']
        if sha256_path(archive) != row['archive_sha256']:
            raise ValueError('Validation archive SHA mismatch')
        if sha256_path(source / f"PM_{row['asset_id']}.json") != row['spec_sha256']:
            raise ValueError('Validation spec SHA mismatch')
    provenance.update({str(test_plan): sha256_path(test_plan), str(base / 'manifest.json'): sha256_path(base / 'manifest.json')})
    result = {
        'version': 1, 'scope': 'local CPU-only source capacity audit; no rendering, training or model/API calls',
        'base': str(base.relative_to(ROOT)), 'source_dataset': download['dataset'],
        'source_dataset_revision': download['dataset_revision'], 'license': download['license'],
        'license_urls': download.get('license_urls', []),
        'counts': {'downloaded': len(conversion['assets']), 'structural_candidates': len(structural),
                   'structural_rejected': len(conversion['assets']) - len(structural), 'fixed_validation': len(validation),
                   'v2_objects': len(used), 'previously_screened_unique': len(previous),
                   'fresh_verified_queue': len(queue), 'new_automatic_pass': 0, 'new_human_pass': 0,
                   'quality_unverified_object_upper_bound': len(used) + len(queue),
                   'currently_certified_objects': len(used), 'currently_certified_pairs_samples': len(used) * 2},
        'historical_screening': historical,
        'structural_rejection_reasons': dict(Counter(r.get('rejection_reason') for r in conversion['assets'] if r['status'] != 'structural_candidate')),
        'exclusion_reasons': dict(Counter(reason for row in excluded for reason in row['reasons'])),
        'audit': {'all_v2_file_hashes_verified': True, 'pool_archive_spec_sha_verified': True,
                  'train_validation_source_model_content_geometry_checked': True,
                  'frozen_test_objects': len(test_record['objects']), 'frozen_test_source_and_geometry_recomputed': True,
                  'frozen_test_content_hash_comparison': 'available plan digests and raw full_model OBJ SHA; source ZIP and converted spec are different representations',
                  'license_scope': 'existing recorded non-commercial research/education terms reused; no new legal or redistribution grant inferred'},
        'limitations': ['Queue capacity is an upper bound before normal-motion, checker, image-quality and real human review.',
                       'Frozen Test metadata provides upstream original IDs but no PartNet source ZIPs; cross-format content equivalence additionally relies on recomputed existing geometry fingerprint.',
                       'Geometry fingerprints use the existing exact six-decimal vertex-set definition, not a near-duplicate similarity guarantee.'],
        'queue': queue, 'excluded': excluded, 'inputs': provenance,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.base.resolve(), args.out.resolve())['counts']))
