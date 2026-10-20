"""Audit new official PartNet sources against every existing protected pool."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from importlib.metadata import version
from pathlib import Path

from scripts.convert_partnet_mobility import _annotation_audit, _scale_audit, _test_identities
from scripts.prepare_obvious_defect_review import ROOT, sha256_path
from turnitover.assets.urdf import UrdfModel
from turnitover.core.serde import to_dict


def audit(base: Path, source: Path, downloads: Path, out: Path):
    if base == out or base in out.parents:
        raise ValueError('Capacity output must be outside the immutable baseline')
    inputs = {}

    def read(path):
        inputs[str(path.resolve())] = sha256_path(path)
        return json.loads(path.read_text())

    old_source = ROOT / 'data/assets/partnet-mobility-trainval-v1'
    old_download = read(ROOT / 'data/partnet-mobility-pilot/download-manifest.json')
    old_conversion = read(old_source / 'conversion.json')
    original_splits = read(old_source / 'splits.json')
    new_download = read(downloads / 'download-manifest.json')
    conversion = read(source / 'conversion.json')
    read(source / 'catalog.json')
    for field in ('dataset', 'dataset_revision', 'license'):
        if new_download[field] != old_download[field] or conversion[field] != old_conversion[field]:
            raise ValueError(f'Official source provenance changed: {field}')
    if new_download['status'] != 'complete' or conversion['status'] != 'complete':
        raise ValueError('Complete download and conversion are required')
    manifest = read(base / 'manifest.json')
    for name, digest in manifest['files'].items():
        if sha256_path(base / name) != digest:
            raise ValueError(f'Immutable baseline SHA mismatch: {name}')
    used = read(base / manifest['source_catalog'])['assets']
    splits = read(base / manifest['expanded_splits'])
    if any(splits['groups'].get(k) != v for k, v in original_splits['groups'].items()):
        raise ValueError('Baseline changed existing split assignments')
    validation = read(base / 'validation.json')
    if validation['training_eligible_objects'] != len(used):
        raise ValueError('Baseline must consist of fully human-qualified pairs')
    supplemental = read(base / 'private/protected_pool_audit.json')
    for name, digest in supplemental['inputs'].items():
        if sha256_path(Path(name)) != digest:
            raise ValueError(f'Protected sibling pool provenance changed: {name}')
        inputs[name] = digest
    keys = ('source_identity', 'model_id', 'archive_sha256', 'spec_sha256', 'geometry_fingerprint')
    protected = {key: set() for key in keys}
    protected['source_identity'].update(f"PartNet-Mobility:{r['asset_id']}" for r in old_download['archives'])
    protected['archive_sha256'].update(r['sha256'] for r in old_download['archives'])
    old_structural = [r for r in old_conversion['assets'] if r['status'] == 'structural_candidate']
    protected_annotations = {r['annotation_id'] for r in old_structural}
    for row in old_structural:
        spec_path = old_source / f"PM_{row['asset_id']}.json"
        if sha256_path(spec_path) != row['spec_sha256']:
            raise ValueError('Protected original source spec SHA mismatch')
        inputs[str(spec_path.resolve())] = row['spec_sha256']
        for key in keys:
            protected[key].add(row['scale_audit']['geometry_fingerprint'] if key == 'geometry_fingerprint' else row[key])
    for row in used:
        for key, field in (('source_identity', 'source_identity'), ('model_id', 'model_id'),
                           ('archive_sha256', 'archive_sha256'), ('spec_sha256', 'original_spec_sha256'),
                           ('geometry_fingerprint', 'original_geometry_fingerprint')):
            if row.get(field):
                protected[key].add(row[field])
    for row in supplemental['protected']:
        protected['source_identity'].add(row['source_identity'])
        if row['source_original_id'] in supplemental['partnet_alias_ids']:
            protected['source_identity'].add('PartNet-Mobility:' + row['source_original_id'])
        protected['model_id'].update(v for v in (row['source_original_id'], row.get('model_id')) if v)
        protected['archive_sha256'].update((row['source_archive_sha256'], row['full_model_sha256']))
        protected['spec_sha256'].add(row['spec_sha256'])
        protected['geometry_fingerprint'].add(row['geometry_fingerprint'])
    test_plan = ROOT / 'configs/uniphys-test.json'
    test_root = ROOT / 'data/assets/uniphys/data'
    test_record = read(test_plan)
    if conversion['formal_test_plan_sha256'] != sha256_path(test_plan):
        raise ValueError('New conversion uses a different frozen Test plan')
    test = _test_identities(test_plan, test_root)
    protected['source_identity'].update(test['source_ids'])
    protected['geometry_fingerprint'].update(test['geometry'])
    for row in test_record['objects']:
        folder = test_root / row['id']
        meta = read(folder / 'meta_data.json')
        model_sha = sha256_path(folder / 'full_model/model.obj')
        if model_sha != row['geometry_sha256']:
            raise ValueError('Frozen Test geometry SHA changed')
        inputs[str((folder / 'full_model/model.obj').resolve())] = model_sha
        protected['archive_sha256'].add(model_sha)
        protected['model_id'].add(str(meta['source']['original_id']))
        if meta['source'].get('model_id'):
            protected['model_id'].add(str(meta['source']['model_id']))
    archives = {r['asset_id']: r for r in new_download['archives']}
    if len(archives) != len(new_download['archives']):
        raise ValueError('Duplicate source IDs in new download')
    queue, excluded = [], []
    seen = {key: set() for key in keys}
    structural = [r for r in conversion['assets'] if r['status'] == 'structural_candidate']
    for row in structural:
        aid = row['asset_id']
        archive = downloads / archives[aid]['path']
        digest = sha256_path(archive)
        if digest != archives[aid]['sha256'] or digest != row['archive_sha256']:
            raise ValueError(f'New source archive SHA mismatch: {aid}')
        spec_path = source / f'PM_{aid}.json'
        spec = read(spec_path)
        if inputs[str(spec_path.resolve())] != row['spec_sha256']:
            raise ValueError(f'New source spec SHA mismatch: {aid}')
        folder = downloads / 'extracted' / aid
        meta = read(folder / 'meta.json')
        if str(meta['anno_id']) != row['annotation_id'] or meta['model_id'] != row['model_id']:
            raise ValueError(f'Original source identity mismatch: {aid}')
        model = UrdfModel(folder / 'mobility.urdf')
        native = model.spec(f'PM_{aid}', meta['model_cat'])
        if json.loads(json.dumps(to_dict(native))) != spec:
            raise ValueError(f'Converted spec differs from native URDF: {aid}')
        if _annotation_audit(folder, native) != row['annotation_audit']:
            raise ValueError(f'Native annotation audit differs: {aid}')
        scale = _scale_audit(folder, model)
        if scale['geometry_fingerprint'] != row['scale_audit']['geometry_fingerprint']:
            raise ValueError(f'Native rest geometry fingerprint differs: {aid}')
        if row.get('frozen_test_overlap') is not False:
            raise ValueError(f'Frozen Test exclusion was not recorded: {aid}')
        record = {key: row[key] for key in ('asset_id', 'source_identity', 'model_id', 'category', 'archive_sha256', 'spec_sha256')}
        record['geometry_fingerprint'] = scale['geometry_fingerprint']
        record['annotation_id'] = row['annotation_id']
        reasons = []
        if row['annotation_id'] in protected_annotations:
            reasons.append('protected_source_overlap_annotation_id')
        for key in keys:
            if record[key] in protected[key]:
                reasons.append('protected_source_overlap_' + key)
            if record[key] in seen[key]:
                reasons.append('new_pool_duplicate_' + key)
        if reasons:
            excluded.append({**record, 'reasons': reasons})
        else:
            queue.append({**record, 'source_split': 'unassigned', 'candidate_split': 'train',
                          'previous_motion_decision': None, 'status': 'pending_normal_motion_and_level0_screening'})
            for key in keys:
                seen[key].add(record[key])
    audit_script = str(Path(__file__).resolve())
    inputs[audit_script] = sha256_path(Path(audit_script))
    counts = {'downloaded': len(archives), 'structural_candidates': len(structural),
              'structural_rejected': len(conversion['assets']) - len(structural), 'base_objects': len(used),
              'fresh_verified_queue': len(queue), 'currently_certified_objects': len(used),
              'currently_certified_pairs_samples': 2 * len(used), 'new_automatic_pass': 0, 'new_human_pass': 0,
              'quality_unverified_object_upper_bound': len(used) + len(queue)}
    result = {'version': 1, 'audit_script': audit_script, 'runtime': {'numpy': version('numpy'), 'trimesh': version('trimesh')}, 'scope': 'new official source CPU audit; no rendering, models or API calls',
              'base': str(base.relative_to(ROOT)), 'source_dataset': new_download['dataset'],
              'source_dataset_revision': new_download['dataset_revision'], 'license': new_download['license'],
              'license_urls': new_download.get('license_urls', []), 'counts': counts,
              'protected_counts': {'all_original_source_ids': len(old_download['archives']),
                                   'original_structural_objects': len(old_structural), 'used_train_objects': len(used),
                                   'fixed_validation': sum(v == 'validation' for v in original_splits['groups'].values()),
                                   'sibling_independent_objects': supplemental['counts']['independent_upstream_objects'],
                                   'frozen_test_objects': len(test_record['objects'])},
              'structural_rejection_reasons': dict(Counter(r.get('rejection_reason') for r in conversion['assets'] if r['status'] != 'structural_candidate')),
              'exclusion_reasons': dict(Counter(reason for r in excluded for reason in r['reasons'])),
              'audit': {'baseline_all_file_shas_verified': True, 'source_revision_license_matched_original': True,
                        'archives_and_specs_sha_verified': True, 'native_urdf_specs_annotations_geometry_recomputed': True,
                        'all_existing_source_model_content_exact_geometry_checked': True},
              'limitations': ['Queue is capacity before automatic image/checker and genuine human review.',
                              'Cross-dataset missing model IDs and coordinate/tessellation changes limit exact geometry equivalence; existing exact fingerprint definition is preserved.'],
              'queue': queue, 'excluded': excluded, 'inputs': inputs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    report = '# Round 2 来源容量审计\n\n' + '\n'.join(f'- {k}: {v}' for k, v in counts.items())
    report += '\n\n仅为来源合格且未重复的待渲染候选上界；尚无新增自动或人工通过样本。来源许可与修订沿用既有官方记录。完整拒绝原因及保护集合见 private/capacity_audit.json。\n'
    (out.parent.parent / 'SOURCE_AUDIT.md').write_text(report)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'docs/evidence/partnet-mobility-level0-axis-v3')
    parser.add_argument('--source', type=Path, default=ROOT / 'data/assets/partnet-mobility-level0-round2')
    parser.add_argument('--downloads', type=Path, default=ROOT / 'data/partnet-mobility-level0-round2')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/evidence/partnet-mobility-level0-axis-v4/private/capacity_audit.json')
    args = parser.parse_args()
    print(json.dumps(audit(args.base.resolve(), args.source.resolve(), args.downloads.resolve(), args.out.resolve())['counts']))
