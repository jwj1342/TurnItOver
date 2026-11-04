"""Import actual review JSONL into a new private directory, retaining raw bytes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

if __package__:
    from .prepare_tier_review import validate as validate_pack
    from .tier_review_records import classify, verify_sets
else:
    from prepare_tier_review import validate as validate_pack
    from tier_review_records import classify, verify_sets


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_return(raw: Path, raw_sha: str, package: Path, package_sha: str,
                  mapping: Path, mapping_sha: str, metadata: Path,
                  metadata_sha: str, output: Path) -> dict:
    if output.exists():
        raise ValueError('Import output must be a new directory')
    for path, expected in ((raw, raw_sha), (package / 'package.json', package_sha),
                           (mapping, mapping_sha), (metadata, metadata_sha)):
        if digest(path) != expected:
            raise ValueError(f'Input SHA differs: {path.name}')
    validate_pack(package)
    raw_rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    custody = json.loads(mapping.read_text())
    cases = json.loads((package / 'case_manifest.json').read_text())
    if custody.get('version') != 1 or custody.get('package_sha256') != package_sha:
        raise ValueError('Custody mapping belongs to a different package')
    verify_sets(raw_rows, custody['mapping'], cases, package_sha)
    pack_files = json.loads((package / 'package.json').read_text())['files']
    if any(pack_files.get(media['path']) != media['sha256']
           for item in custody['mapping'] for media in item['media']):
        raise ValueError('Custody mapping media SHA differs from reviewed pack')
    metadata_rows = json.loads(metadata.read_text())
    if not isinstance(metadata_rows, list) or len({row['case_id'] for row in metadata_rows}) != len(metadata_rows):
        raise ValueError('Case metadata must be a list with unique case IDs')
    by_id = {row['case_id']: row for row in metadata_rows}
    raw_by_id = {row['case_id']: row for row in raw_rows}
    normalized = []
    seen = set()
    for item in custody['mapping']:
        cid = item['source_case_id']
        if cid in seen or cid not in by_id:
            raise ValueError('Missing or ambiguous source case metadata')
        seen.add(cid)
        meta = by_id[cid]
        if str(Path(meta['bundle']).resolve()) != str(Path(item['bundle']).resolve()):
            raise ValueError('Case metadata bundle disagrees with custody mapping')
        if meta.get('tier') not in ('easy', 'medium', 'hard') or meta.get('kind') not in ('clean', 'defect'):
            raise ValueError('Case metadata has invalid expected tier/kind')
        status, review = classify(raw_by_id[item['case_id']], meta)
        normalized.append((cid, item['case_id'], status, review))
    if seen != set(by_id):
        raise ValueError('Metadata has cases outside the reviewed package')
    output.mkdir(parents=True)
    try:
        shutil.copyfile(raw, output / 'raw-return.jsonl')
        shutil.copyfile(mapping, output / 'mapping.json')
        shutil.copyfile(metadata, output / 'case-metadata.json')
        shutil.copytree(package, output / 'review-pack')
        references = {key: {'path': name, 'sha256': digest(output / name)} for key, name in (
            ('raw_return', 'raw-return.jsonl'), ('custody_mapping', 'mapping.json'),
            ('case_metadata', 'case-metadata.json'), ('blind_package', 'review-pack/package.json'))}
        for index, (cid, opaque, status, review) in enumerate(normalized):
            document = dict(version=1, case_id=cid, opaque_case_id=opaque, import_status=status,
                review_package_sha256=package_sha, difficulty_review=review, **references)
            (output / f'return-{index:04d}.json').write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n')
        report = {'version': 1, 'cases': len(normalized),
                  'counts': dict(Counter(item[2] for item in normalized)),
                  'raw_return_sha256': raw_sha, 'package_sha256': package_sha,
                  'human_identity_authenticated': False}
        (output / 'import-report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
    except BaseException:
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'package', 'mapping', 'metadata', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('raw', 'package', 'mapping', 'metadata'):
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(import_return(args.raw, args.raw_sha256, args.package, args.package_sha256,
        args.mapping, args.mapping_sha256, args.metadata, args.metadata_sha256, args.output), indent=2))


if __name__ == '__main__':
    main()
