"""Supplement Level 0 provenance with sibling formal UniPhys Train/Validation pools."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import trimesh

from scripts.convert_partnet_mobility import _geometry_fingerprint
from scripts.prepare_obvious_defect_review import ROOT, sha256_path


def audit(dataset: Path, protected_root: Path):
    capacity = dataset / 'private/capacity_audit.json'
    queue = json.loads(capacity.read_text())['queue']
    inputs = {str(capacity): sha256_path(capacity)}
    protected, aliases, overlaps = [], set(), []
    counts = Counter()
    for catalog_path in sorted((protected_root / 'assets').glob('uniphys40k-trainval-v*/catalog.json')):
        split_path = catalog_path.parent / 'splits.json'
        catalog = json.loads(catalog_path.read_text())
        splits = json.loads(split_path.read_text())['groups']
        inputs.update({str(p.resolve()): sha256_path(p) for p in (catalog_path, split_path)})
        for row in catalog['assets']:
            spec_path = catalog_path.parent / row['spec']
            digest = sha256_path(spec_path)
            if row.get('spec_sha256', digest) != digest:
                raise ValueError(f'Protected spec SHA mismatch: {spec_path}')
            inputs[str(spec_path.resolve())] = digest
            spec = json.loads(spec_path.read_text())
            aid = spec['asset_id']
            archives = list((protected_root / 'uniphys40k-pilot').glob(f'part_*/{aid}.zip'))
            if len(archives) != 1:
                raise ValueError(f'Expected one source ZIP for protected object: {aid}')
            archive = archives[0]
            archive_sha = sha256_path(archive)
            if row.get('source_content_sha256', archive_sha) != archive_sha:
                raise ValueError(f'Protected archive SHA mismatch: {aid}')
            inputs[str(archive.resolve())] = archive_sha
            with zipfile.ZipFile(archive) as zf:
                meta_name = next(n for n in zf.namelist() if n.endswith('/meta_data.json'))
                mesh_name = next(n for n in zf.namelist() if n.endswith('/full_model/model.obj'))
                metadata = json.loads(zf.read(meta_name))
                mesh_bytes = zf.read(mesh_name)
            upstream = metadata['source']
            original_id = str(upstream['original_id'])
            identity = f"{upstream['dataset']}:{original_id}"
            if identity not in row['split_group']:
                raise ValueError(f'Catalog/upstream identity mismatch: {aid}')
            if upstream['dataset'].lower() in ('partnet', 'partnet-mobility', 'partnetmobility'):
                aliases.add(original_id)
            scene = trimesh.load(io.BytesIO(mesh_bytes), file_type='obj', force='scene', process=False)
            vertices = np.concatenate([m.vertices for m in scene.geometry.values()])
            fp = _geometry_fingerprint(vertices)
            mesh_sha = hashlib.sha256(mesh_bytes).hexdigest()
            split = splits[row['split_group']]
            if split not in ('train', 'validation'):
                raise ValueError('Unexpected protected pool split')
            record = {'catalog': str(catalog_path.resolve()), 'asset_id': aid, 'split': split,
                      'source_identity': identity, 'source_original_id': original_id,
                      'model_id': upstream.get('model_id'), 'source_archive_sha256': archive_sha,
                      'spec_sha256': digest, 'full_model_sha256': mesh_sha, 'geometry_fingerprint': fp}
            protected.append(record)
            counts[split] += 1
            for candidate in queue:
                reasons = []
                if upstream['dataset'].lower() in ('partnet', 'partnet-mobility', 'partnetmobility') and candidate['asset_id'] == original_id:
                    reasons.append('partnet_source_identity_alias')
                if candidate['model_id'] in (original_id, upstream.get('model_id')):
                    reasons.append('source_model_id')
                if candidate['archive_sha256'] in (archive_sha, mesh_sha) or candidate['spec_sha256'] == digest:
                    reasons.append('content_sha256')
                if candidate['geometry_fingerprint'] == fp:
                    reasons.append('exact_geometry_fingerprint')
                if reasons:
                    overlaps.append({'candidate': candidate['asset_id'], 'protected': aid, 'reasons': reasons})
    if not protected:
        raise ValueError('No protected formal pools found')
    result = {'version': 1, 'scope': 'read-only CPU supplemental audit; no asset mutations or rendering',
              'counts': {'catalog_records': len(protected), 'independent_upstream_objects': len({r['source_identity'] for r in protected}),
                         'queue_objects': len(queue), 'overlaps': len(overlaps), 'split_records': dict(counts)},
              'partnet_alias_ids': sorted(aliases), 'overlaps': overlaps, 'protected': protected,
              'limitations': ['UniPhys upstream records lack independent ShapeNet model_id fields; original_id and available explicit model_id are compared.',
                              'ZIP/spec/content and six-decimal raw vertex fingerprints check exact representations; normalization, rotation or different tessellation may evade an exact geometry fingerprint.'],
              'inputs': inputs}
    out = dataset / 'private/protected_pool_audit.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    return result['counts']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--protected-root', type=Path, default=ROOT.parent / 'TurnItOver/data')
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset.resolve(), args.protected_root.resolve())))
