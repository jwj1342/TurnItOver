"""SHA-pinned local sources for new Medium objects; historical bundles stay read-only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from turnitover.assets.catalog import decode_spec


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(value):
    """Extract source IDs recursively from private exclusion ledgers."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('source_identity', 'split_group') and isinstance(item, str):
                found.add(item)
            elif key == 'asset_id' and isinstance(item, str) and item.startswith('PM_'):
                found.add('PartNet-Mobility:' + item[3:])
            else:
                found.update(identities(item))
    elif isinstance(value, list):
        for item in value:
            found.update(identities(item))
    return found


def load_exclusions(paths):
    excluded, pins, fingerprints = set(), {}, set()
    for path in map(Path, paths):
        pins[str(path.resolve())] = sha(path)
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                names = [name for name in archive.namelist() if name.endswith('/private/gold.jsonl')]
                if len(names) != 1:
                    raise ValueError(f'Expected one private Gold ledger: {path}')
                rows = [json.loads(line) for line in archive.read(names[0]).splitlines() if line.strip()]
        else:
            text = path.read_text()
            rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
                    if path.suffix == '.jsonl' else json.loads(text))
        excluded.update(identities(rows))
        def collect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ("original_geometry_fingerprint", "geometry_fingerprint") and isinstance(item, str):
                        fingerprints.add(item)
                    else:
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
        collect(rows)
    return excluded, pins, fingerprints


def load_sources(pairs, exclusions, splits_path):
    excluded, pins, blocked_geometry = load_exclusions(exclusions)
    splits_path = Path(splits_path).resolve()
    pins[str(splits_path)] = sha(splits_path)
    splits = json.loads(splits_path.read_text())
    pool, seen, geometry = [], set(), set()
    for catalog_path, download_path in pairs:
        catalog_path, download_path = Path(catalog_path).resolve(), Path(download_path).resolve()
        conversion_path = catalog_path.parent / 'conversion.json'
        for path in (catalog_path, conversion_path, download_path):
            pins[str(path)] = sha(path)
        catalog, conversion, download = [json.loads(p.read_text()) for p in
                                         (catalog_path, conversion_path, download_path)]
        if download['dataset_revision'] != conversion['dataset_revision']:
            raise ValueError('Source revision mismatch')
        entries = {Path(row['spec']).stem: row for row in catalog['assets']}
        archives = {str(row['asset_id']): row for row in download['archives']}
        for row in conversion['assets']:
            identity = row['source_identity']
            if (row['status'] != 'structural_candidate' or row.get('frozen_test_overlap') is not False
                    or identity in excluded or identity in seen
                    or splits['groups'].get(identity) in ('validation', 'test')):
                continue
            aid = 'PM_' + str(row['asset_id'])
            spec_path = catalog_path.parent / (aid + '.json')
            raw = json.loads(spec_path.read_text())
            if len(raw['joints']) < 2:
                continue
            fingerprint = row['scale_audit']['geometry_fingerprint']
            if fingerprint in geometry or fingerprint in blocked_geometry:
                continue
            spec = decode_spec(raw)
            archive_path = download_path.parent / archives[str(row['asset_id'])]['path']
            urdf = download_path.parent / 'extracted' / str(row['asset_id']) / 'mobility.urdf'
            for path, digest in ((spec_path, row['spec_sha256']), (archive_path, row['archive_sha256'])):
                if sha(path) != digest:
                    raise ValueError(f'Source SHA mismatch: {path}')
                pins[str(path)] = digest
            pins[str(urdf)] = sha(urdf)
            metadata = dict(entries.get(aid, {}))
            metadata.update(source_identity=identity, split_group=identity,
                model_id=row['model_id'], source_asset_id=str(row['asset_id']),
                dataset=download['dataset'], dataset_revision=download['dataset_revision'],
                source_dataset=download['dataset'], license=download['license'],
                source_spec=str(spec_path), source_archive=str(archive_path), source_catalog=str(catalog_path),
                archive_sha256=row['archive_sha256'], original_spec_sha256=row['spec_sha256'],
                original_geometry_fingerprint=fingerprint, formal_training=False,
                upstream_structural_checks={'scale_audit': row['scale_audit'],
                    'annotation_audit': row['annotation_audit'], 'frozen_test_overlap': False})
            pool.append(dict(spec=spec, metadata=metadata, source_urdf=str(urdf),
                complexity={'active_joints': len(spec.joints), 'parts': len(spec.parts), 'category': spec.category}))
            seen.add(identity)
            geometry.add(fingerprint)
    pool.sort(key=lambda item: hashlib.sha256(('medium-expansion-v1:' +
                                               item['spec'].asset_id).encode()).hexdigest())
    return pool, pins
