"""A complete root-relative inventory, including nested review manifests."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

if __package__:
    from .easy_release_archive import sha256
else:
    from easy_release_archive import sha256

NAME = 'bundle_manifest.json'


def _files(root):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Medium bundle must not contain symbolic links')
    return {p.relative_to(root).as_posix(): p for p in root.rglob('*')
            if p.is_file() and p.relative_to(root).as_posix() != NAME}


def write_manifest(root: Path):
    files = {name: sha256(path) for name, path in sorted(_files(root).items())}
    (root / NAME).write_text(json.dumps({'version': 1, 'files': files}, indent=2, sort_keys=True) + '\n')
    return files


def validate_manifest(root: Path):
    encoded = (root / NAME).read_text()
    manifest = json.loads(encoded)
    if (not isinstance(manifest, dict) or set(manifest) != {'version', 'files'}
            or manifest.get('version') != 1 or not isinstance(manifest.get('files'), dict)):
        raise ValueError('Invalid Medium bundle inventory')
    if encoded != json.dumps(manifest, indent=2, sort_keys=True) + '\n':
        raise ValueError('Medium inventory must use its canonical encoding without hidden fields')
    actual = _files(root)
    if set(actual) != set(manifest['files']):
        raise ValueError('Medium inventory has missing or unlisted files')
    for name, expected in manifest['files'].items():
        if (not isinstance(name, str) or '\\' in name or PurePosixPath(name).is_absolute()
                or any(p in ('', '.', '..') for p in name.split('/')) or sha256(actual[name]) != expected):
            raise ValueError(f'Invalid path or SHA-256 mismatch: {name}')
    return set(actual)
