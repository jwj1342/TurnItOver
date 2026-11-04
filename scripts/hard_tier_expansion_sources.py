"""Screen reserved official conversions with the unchanged Hard structural policy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from hard_tier_expansion_artifacts import read, sha, write


def prepare(builder_root, reservations, out):
    sys.path.insert(0, str(builder_root.resolve()))
    from scripts.hard_candidate_policy import candidate_targets, load_semantics, criteria
    from turnitover.assets.catalog import decode_spec
    from turnitover.assets.urdf import UrdfModel
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Source preparation requires a new output directory')
    out.mkdir(parents=True)
    accepted, rejected = [], []
    inputs = {str(reservations.resolve()): sha(reservations), str(Path(__file__).resolve()): sha(__file__)}
    official_path = Path(__file__).resolve().parents[1] / 'output/hard-official-archive-verification-v1.json'
    official = read(official_path) if official_path.exists() else None
    official_by_id = {r['asset_id']: r for r in official['archives']} if official else {}
    if official:
        inputs[str(official_path)] = sha(official_path)
    policy_file = builder_root.resolve() / 'scripts/hard_candidate_policy.py'
    inputs[str(policy_file)] = sha(policy_file)
    for row in read(reservations)['assets']:
        record, spec_path, urdf = row['conversion_record'], Path(row['source_spec']), Path(row['source_urdf'])
        if record['status'] != 'structural_candidate' or record['frozen_test_overlap']:
            raise ValueError('Conversion eligibility failed')
        if sha(spec_path) != record['spec_sha256']:
            raise ValueError('Converted spec SHA mismatch')
        spec = decode_spec(read(spec_path))
        targets, reasons = candidate_targets(spec, UrdfModel(urdf), load_semantics(urdf))
        targets = [t for t in targets if min(t['original_limits']) <= 0 <= max(t['original_limits'])
                   and (t['joint_type'] != 'revolute' or max(abs(v) for v in t['original_limits']) <= 3.141592663589793)]
        if not targets:
            rejected.append({'asset_id': row['asset_id'], 'reasons': reasons or ['no_legal_full_motion_target']})
            continue
        manifest_path = Path(row['download_manifest'])
        manifest = read(manifest_path)
        archive_row = next(r for r in manifest['archives'] if str(r['asset_id']) == str(record['asset_id']))
        archive = manifest_path.parent / archive_row['path']
        verified = official_by_id.get(str(record['asset_id']))
        if verified:
            if verified['sha256'] != record['archive_sha256'] or verified['path'] != str(archive.resolve()):
                raise ValueError('Official LFS verification disagrees with candidate source')
        elif archive_row['sha256'] != record['archive_sha256'] or sha(archive) != record['archive_sha256']:
            raise ValueError('Official archive provenance mismatch')
        metadata = {'dataset': manifest['dataset'], 'source_dataset': manifest['dataset'],
            'dataset_revision': manifest['dataset_revision'], 'license': manifest['license'],
            'source_identity': row['source_identity'], 'source_asset_id': str(record['asset_id']),
            'model_id': record['model_id'], 'archive_sha256': record['archive_sha256'],
            'original_spec_sha256': record['spec_sha256'],
            'original_geometry_fingerprint': record['scale_audit']['geometry_fingerprint'],
            'source_spec': str(spec_path.resolve()), 'source_archive': str(archive.resolve())}
        files = {str(p.resolve()): sha(p) for p in urdf.parent.rglob('*') if p.is_file()}
        files[str(spec_path.resolve())] = record['spec_sha256']
        source_record = out / 'qualified' / (row['asset_id'] + '.json')
        write(source_record, {'asset_id': row['asset_id'], 'metadata': metadata, 'files': files,
                             'conversion_record': record})
        accepted.append({'asset_id': row['asset_id'], 'source_identity': row['source_identity'],
            'category': spec.category, 'active_joints': len(spec.joints), 'parts': len(spec.parts),
            'source_spec': str(spec_path.resolve()), 'source_urdf': str(urdf.resolve()),
            'source_record': str(source_record), 'preferred_target': targets[0], 'eligible_targets': targets,
            'qualification': 'structural candidate only; render and human review pending'})
        inputs[str(manifest_path.resolve())] = sha(manifest_path)
    write(out / 'pool.json', accepted)
    write(out / 'rejections.json', rejected)
    write(out / 'manifest.json', {'criteria': criteria(), 'inputs': inputs,
        'accepted': len(accepted), 'rejected': len(rejected),
        'files': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}})
    return {'accepted': len(accepted), 'rejected': len(rejected)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builder-root', type=Path, required=True)
    parser.add_argument('--reservations', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.builder_root, args.reservations, args.out)))
