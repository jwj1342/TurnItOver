"""Read hash-pinned custody inventories and reject frozen geometry reuse."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

if __package__:
    from .tier_expansion_evidence import _file
else:
    from tier_expansion_evidence import _file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pins(root: Path, manifest: dict, contract: dict) -> None:
    expected = contract.get('custody_pins_sha256')
    reference = manifest.get('custody_pins')
    if not expected or not isinstance(reference, dict) or reference.get('sha256') != expected:
        raise ValueError('Missing or incorrect trusted custody pins')
    path = _file(root, reference)
    pins = json.loads(path.read_text())
    if pins.get('version') != 1 or pins.get('algorithm') != 'conversion-scale-audit-unique-rounded-world-vertices-float64-v1':
        raise ValueError('Unsupported custody geometry algorithm')
    for entry in pins['protected_files']:
        original = Path(entry['path'])
        if not original.is_file() or sha256(original) != entry['sha256']:
            raise ValueError(f'Protected baseline/v2 bytes changed: {original}')
    # Also detect added/deleted files, rather than checking only surviving paths.
    frozen_root = Path(pins['frozen_root'])
    expected_names = set(pins['frozen_files'])
    actual_names = {p.relative_to(frozen_root).as_posix() for p in frozen_root.rglob('*') if p.is_file()}
    if actual_names != expected_names:
        raise ValueError('Frozen v2 file inventory changed')
    for archive in pins['baseline_archives']:
        with zipfile.ZipFile(archive['path']) as zf:
            actual = {item.filename: hashlib.sha256(zf.read(item)).hexdigest()
                      for item in zf.infolist() if not item.is_dir()}
        if actual != archive['members']:
            raise ValueError('Baseline archive file inventory changed')
    frozen_ids = {item['source_identity'] for item in pins['frozen_objects']}
    frozen_geometry = {item['geometry_fingerprint'] for item in pins['frozen_objects']}
    sources = pins['source_bindings']
    baselines = pins['baseline_objects']
    for row in manifest['objects']:
        source = row['source_identity']
        if row['membership'] == 'baseline':
            trusted = baselines.get(row['tier'], {}).get(source)
            if trusted is None or row['geometry_fingerprint'] != trusted['geometry_fingerprint']:
                raise ValueError('Baseline geometry differs from trusted source inventory')
        else:
            spec_reference = row.get('original_spec')
            spec = _file(root, spec_reference)
            candidates = sources.get(source, [])
            matches = [item for item in candidates if item['spec_sha256'] == sha256(spec)]
            if not matches or any(item['geometry_fingerprint'] != row['geometry_fingerprint'] for item in matches):
                raise ValueError('Original spec/geometry lacks trusted conversion binding')
            for item in matches:
                for field, digest in (('spec_path', 'spec_sha256'), ('archive_path', 'archive_sha256')):
                    original = Path(item[field])
                    if not original.is_file() or sha256(original) != item[digest]:
                        raise ValueError('Trusted original spec/archive bytes changed')
        if any(case['membership'] == 'added' for case in row['cases']):
            if source in frozen_ids or row['geometry_fingerprint'] in frozen_geometry:
                raise ValueError('Added case reuses frozen evaluated v2 identity or geometry')
