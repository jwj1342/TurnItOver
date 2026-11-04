"""Seal explicitly selected private candidate/review trees as immutable ZIPs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.prepare_tier_review import validate


def seal(roots: list[Path], output: Path, review: bool = False) -> dict:
    if output.exists() or not roots or len({p.name for p in roots}) != len(roots):
        raise ValueError('New output and distinct named source roots are required')
    snapshots = []
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError('Source must be a regular directory')
        if review:
            validate(root)
        files = sorted(p for p in root.rglob('*') if p.is_file())
        for p in files:
            if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
                raise ValueError('Symbolic/escaping candidate artifacts are not allowed')
            snapshots.append((p, root.name + '/' + p.relative_to(root).as_posix(),
                              hashlib.sha256(p.read_bytes()).hexdigest()))
    output.mkdir(parents=True)
    try:
        archive = output / ('review.zip' if review else 'private-candidates.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for path, name, digest in snapshots:
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError('Input changed while sealing: ' + str(path))
                info = zipfile.ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                z.writestr(info, data)
        for path, _, digest in snapshots:
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Input changed while sealing: ' + str(path))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        manifest = {'version': 1, 'status': 'pending_human_review', 'training_ready': False,
                    'archive': archive.name, 'sha256': digest, 'bytes': archive.stat().st_size,
                    'files': {name: sha for _, name, sha in snapshots}}
        (output / 'snapshot.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        (output / 'SHA256SUMS').write_text(digest + '  ' + archive.name + '\n')
        return {'archive': str(archive), 'sha256': digest, 'files': len(snapshots)}
    except BaseException:
        import shutil
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(seal(args.root, args.output, args.review_only), sort_keys=True))


if __name__ == '__main__':
    main()
