"""Acquire a SHA-pinned official PartNet source slice into a new private tree."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

REVISION = 'ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f'
REPO = 'sapien-sim/PartNetMobility'


def main():
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--min-archive-mb', type=int, default=0)
    parser.add_argument('--max-archive-mb', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.out.exists() or args.count < 1 or args.skip < 0:
        parser.error('Use a new output, positive count and nonnegative skip')
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    eligible = [r for r in info.siblings if re.fullmatch(r'\d+\.zip', r.rfilename)
                and r.size is not None
                and args.min_archive_mb * 1024 ** 2 < r.size <= args.max_archive_mb * 1024 ** 2]
    eligible.sort(key=lambda r: hashlib.sha256(f'{args.seed}:{r.rfilename}'.encode()).digest())
    selected = eligible[args.skip:args.skip + args.count]
    if len(selected) != args.count:
        raise ValueError('Slice exceeds available sources')
    args.out.mkdir(parents=True)
    def fetch(row):
        path = Path(hf_hub_download(REPO, row.rfilename, repo_type='dataset',
                                   revision=info.sha, local_dir=args.out))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        lfs = getattr(row, 'lfs', None)
        expected = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        if path.stat().st_size != row.size or (expected is not None and expected != sha):
            raise ValueError('Official archive size or SHA mismatch: ' + row.rfilename)
        return {'asset_id': path.stem, 'path': row.rfilename, 'bytes': row.size, 'sha256': sha}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        archives = list(executor.map(fetch, selected))
    manifest = {'version': 1, 'status': 'complete', 'dataset': REPO, 'dataset_revision': info.sha,
                'license': 'PartNet-Mobility terms: non-commercial research/education; ShapeNet terms also apply',
                'license_urls': ['https://huggingface.co/datasets/sapien-sim/PartNetMobility',
                                 'https://huggingface.co/datasets/ShapeNet/ShapeNetCore'],
                'selection': {'method': 'sha256(seed:filename)', 'seed': args.seed, 'skip': args.skip,
                              'count': args.count, 'min_archive_mb_exclusive': args.min_archive_mb,
                              'max_archive_mb': args.max_archive_mb},
                'downloaded_at_utc': datetime.now(timezone.utc).isoformat(), 'archives': archives}
    (args.out / 'download-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'archives': len(archives), 'manifest': str(args.out / 'download-manifest.json')}))


if __name__ == '__main__':
    main()
