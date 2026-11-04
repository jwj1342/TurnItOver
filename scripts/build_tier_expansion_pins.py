"""Seal local custody inputs without changing original baselines or evaluated v2."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

if __package__:
    from .tier_expansion_pins import sha256
else:
    from tier_expansion_pins import sha256


def build(workspace: Path, output: Path, catalog_roots=(), download_manifests=(), archive_verifications=()) -> dict:
    if output.exists():
        raise ValueError('Pins output must not exist')
    for path in download_manifests:
        if not Path(path).is_file():
            raise ValueError(f'Explicit download manifest is missing: {path}')
    for path in catalog_roots:
        if not (Path(path) / 'conversion.json').is_file():
            raise ValueError(f'Explicit conversion catalog is missing: {path}')
    for path in archive_verifications:
        if not Path(path).is_file():
            raise ValueError(f'Explicit official verification report is missing: {path}')
    dataset = workspace / 'TurnItOver-dataset'
    release = workspace / 'TurnItOver-dataset-release'
    hard = workspace / 'TurnItOver-hard-pilot'
    frozen = dataset / 'docs/evidence/partnet-mobility-level0-axis-v2'
    protected, bindings = {}, {}
    def protect(path):
        path = path.resolve()
        protected[str(path)] = sha256(path)
    algorithm_source = dataset / 'scripts/convert_partnet_mobility.py'
    protect(algorithm_source)
    for folder in ('turnitover/assets', 'turnitover/core'):
        for path in sorted((dataset / folder).glob('*.py')):
            protect(path)
    # Conversion records are trusted geometry mappings from the existing scale
    # audit, bound to actual original spec and upstream archive bytes.
    downloads = list((dataset / 'data').glob('*/download-manifest.json'))
    downloads += list((hard / 'output').glob('hard-new-download-v*/download-manifest.json'))
    downloads += list(download_manifests)
    archives = {}
    for path in downloads:
        if not path.exists():
            continue
        protect(path)
        for item in json.loads(path.read_text())['archives']:
            archive = path.parent / item['path']
            if archive.is_file():
                archives[item['sha256']] = archive
    for path in archive_verifications:
        report = json.loads(Path(path).read_text())
        if (report.get('kind') != 'current_archive_verification'
                or report.get('dataset_revision') != 'ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f'
                or report.get('not_historical_download_manifest') is not True):
            raise ValueError('Unexpected official archive verification schema or revision')
        protect(Path(path))
        verification_script = release / 'scripts/verify_hard_archive_provenance.py'
        if sha256(verification_script) != report.get('script_sha256'):
            raise ValueError('Official verification script SHA differs from report')
        protect(verification_script)
        for item in report['archives']:
            archive = Path(item['path'])
            record, spec = Path(item['qualified_record']), Path(item['source_spec'])
            if (item['status'] != 'verified_against_official_lfs'
                    or item['sha256'] != item['official_lfs_sha256']
                    or sha256(archive) != item['sha256'] or archive.stat().st_size != item['bytes']
                    or sha256(record) != item['qualified_record_sha256']
                    or sha256(spec) != item['original_spec_sha256']):
                raise ValueError('Official archive verification binding differs')
            archives[item['sha256']] = archive
    conversions = list((dataset / 'data/assets').glob('*/conversion.json'))
    conversions += [Path(path) / 'conversion.json' for path in catalog_roots]
    for path in conversions:
        if not path.exists():
            continue
        document = json.loads(path.read_text())
        if document.get('status') != 'complete':
            raise ValueError(f'Incomplete conversion: {path}')
        protect(path)
        for item in document['assets']:
            if item.get('status') != 'structural_candidate':
                continue
            spec = path.parent / f"PM_{item['asset_id']}.json"
            archive = archives.get(item['archive_sha256'])
            if not spec.is_file() or sha256(spec) != item['spec_sha256'] or archive is None:
                raise ValueError(f'Missing or changed converted source: {item["asset_id"]}')
            if sha256(archive) != item['archive_sha256']:
                raise ValueError('Original archive differs')
            binding = dict(spec_path=str(spec.resolve()), spec_sha256=item['spec_sha256'],
                           archive_path=str(archive.resolve()), archive_sha256=item['archive_sha256'],
                           geometry_fingerprint=item['scale_audit']['geometry_fingerprint'])
            bindings.setdefault(item['source_identity'], []).append(binding)
    unbound_qualified = []
    for path in (hard / 'output').glob('hard-new-download-v*/qualified/*.json'):
        item = json.loads(path.read_text()); meta = item['metadata']
        spec, archive = Path(meta['source_spec']), Path(meta['source_archive'])
        if meta['archive_sha256'] not in archives:
            unbound_qualified.append(str(path))
            continue  # Never admit an unbound source into the trusted inventory.
        if sha256(spec) != meta['original_spec_sha256'] or sha256(archive) != meta['archive_sha256']:
            raise ValueError('Hard original spec/archive differs')
        protect(path)
        bindings.setdefault(meta['source_identity'], []).append(dict(
            spec_path=str(spec), spec_sha256=meta['original_spec_sha256'],
            archive_path=str(archive), archive_sha256=meta['archive_sha256'],
            geometry_fingerprint=meta['original_geometry_fingerprint']))
    frozen_files = {}
    for path in sorted(frozen.rglob('*')):
        if path.is_file():
            protect(path); frozen_files[path.relative_to(frozen).as_posix()] = sha256(path)
    frozen_objects = []
    catalog = json.loads((frozen / 'private/source_catalog/catalog.json').read_text())
    for item in catalog['assets']:
        matches = [record for record in bindings.get(item['source_identity'], [])
                   if record['spec_sha256'] == item['original_spec_sha256']
                   and record['archive_sha256'] == item['archive_sha256']
                   and record['geometry_fingerprint'] == item['original_geometry_fingerprint']]
        if not matches:
            raise ValueError('Frozen v2 original spec/archive/geometry binding cannot be verified')
        frozen_objects.append(dict(source_identity=item['source_identity'],
            geometry_fingerprint=item['original_geometry_fingerprint'],
            original_spec_sha256=item['original_spec_sha256'], archive_sha256=item['archive_sha256']))
    baseline_archives, baseline_objects = [], {}
    for tier, rc in [('easy', 2), ('medium', 1), ('hard', 1)]:
        tag = f'{tier}-candidate-v0.1.0-rc.{rc}'
        path = workspace / f'TIO-datasets/releases/{tag}/{tag}-data.zip'
        contract = json.loads((release / f'benchmark/releases/{tag}.json').read_text())
        if sha256(path) != contract['assets'][path.name]['sha256']:
            raise ValueError('Baseline differs from public release contract')
        protect(path)
        with zipfile.ZipFile(path) as zf:
            members = {item.filename: hashlib.sha256(zf.read(item)).hexdigest()
                       for item in zf.infolist() if not item.is_dir()}
            gold = [json.loads(line) for line in zf.read(f'{tier}/private/gold.jsonl').decode().splitlines() if line.strip()]
        baseline_archives.append(dict(path=str(path), members=members))
        baseline_objects[tier] = {}
        for row in gold:
            source = row.get('source', row)
            identity = source['source_identity']
            matches = [x for x in bindings.get(identity, []) if x['spec_sha256'] == source['original_spec_sha256']]
            if not matches:
                raise ValueError(f'Baseline original spec has no trusted binding: {identity}')
            fingerprints = {x['geometry_fingerprint'] for x in matches}
            if len(fingerprints) != 1:
                raise ValueError('Conflicting trusted fingerprints')
            baseline_objects[tier][identity] = dict(geometry_fingerprint=next(iter(fingerprints)))
    result = dict(version=1, algorithm='conversion-scale-audit-unique-rounded-world-vertices-float64-v1',
        algorithm_source=dict(path=str(algorithm_source), sha256=sha256(algorithm_source)),
        excluded_unbound_qualified_records=unbound_qualified,
        frozen_root=str(frozen), frozen_files=frozen_files, frozen_objects=frozen_objects,
        baseline_archives=baseline_archives, baseline_objects=baseline_objects, source_bindings=bindings,
        protected_files=[dict(path=p, sha256=d) for p,d in sorted(protected.items())])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return dict(path=str(output), sha256=sha256(output), source_identities=len(bindings),
                frozen_objects=len(frozen_objects), protected_files=len(protected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--catalog-root', type=Path, action='append', default=[],
                        help='Additional completed conversion catalog directory; repeat per wave')
    parser.add_argument('--download-manifest', type=Path, action='append', default=[],
                        help='Additional official download manifest; repeat per wave')
    parser.add_argument('--archive-verification', type=Path, action='append', default=[],
                        help='Current official LFS verification report for legacy archives; repeatable')
    args = parser.parse_args()
    print(json.dumps(build(args.workspace, args.output, args.catalog_root,
                           args.download_manifest, args.archive_verification), indent=2))
