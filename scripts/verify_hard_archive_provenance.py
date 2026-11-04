"""Verify current local Hard archives against official SHA-pinned HF LFS metadata.

This creates a verification report at the current time, never a historical
acquisition record. Qualified records and original acquisition trees are read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(roots, out):
    if out.exists():
        raise ValueError('Verification report must use a new path')
    qualified = [p for root in roots for p in sorted(root.glob('qualified/*.json'))]
    documents = [(p, json.loads(p.read_text())) for p in qualified]
    revisions = {r['metadata']['dataset_revision'] for _, r in documents}
    if len(revisions) != 1:
        raise ValueError('Expected one official frozen revision')
    revision = revisions.pop()
    info = HfApi().dataset_info('sapien-sim/PartNetMobility', revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError('Official revision resolution changed')
    official = {r.rfilename: r for r in info.siblings}
    records = []
    for path, row in documents:
        meta = row['metadata']
        archive = Path(meta['source_archive'])
        filename = meta['source_asset_id'] + '.zip'
        entry = official[filename]
        if entry.lfs is None:
            raise ValueError('Official source archive has no LFS SHA')
        expected = entry.lfs.sha256
        actual = sha(archive)
        if actual != expected or actual != meta['archive_sha256'] or archive.stat().st_size != entry.size:
            raise ValueError('Archive differs from official revision: ' + filename)
        if sha(meta['source_spec']) != meta['original_spec_sha256']:
            raise ValueError('Qualified spec SHA mismatch: ' + str(path))
        records.append({'source_identity': meta['source_identity'], 'asset_id': meta['source_asset_id'],
            'qualified_record': str(path.resolve()), 'qualified_record_sha256': sha(path),
            'source_spec': meta['source_spec'], 'original_spec_sha256': meta['original_spec_sha256'],
            'path': str(archive.resolve()), 'bytes': entry.size, 'sha256': actual,
            'official_filename': filename, 'official_lfs_sha256': expected,
            'status': 'verified_against_official_lfs'})
    report = {'version': 1, 'status': 'verified', 'kind': 'current_archive_verification',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'sapien-sim/PartNetMobility', 'dataset_revision': revision,
        'verification_method': 'HF dataset_info(files_metadata=True), exact revision, official LFS SHA256 and bytes',
        'not_historical_download_manifest': True, 'archives': records,
        'script_sha256': sha(__file__)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    return len(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.root, args.out))
