"""Fully verify downloaded Medium assets before publishing their extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

if __package__:
    from .easy_release_archive import extract_zip, sha256, verify_asset_directory
    from .prepare_medium_review import validate_review_bundle
    from .validate_medium_bundle import validate_bundle
else:
    from easy_release_archive import extract_zip, sha256, verify_asset_directory
    from prepare_medium_review import validate_review_bundle
    from validate_medium_bundle import validate_bundle


def validate_release(assets: Path, manifest: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError('Extraction output must not exist')
    contract = verify_asset_directory(assets, manifest, {'medium', 'medium_review'})
    if contract.get('dataset') != 'medium' or contract.get('training_ready') is not False:
        raise ValueError('Expected a non-training Medium candidate contract')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.medium-release-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'validated'
        staging.mkdir()
        for name, entry in contract['assets'].items():
            extract_zip(assets / name, staging, entry)
        root = staging / 'medium'
        report = validate_bundle(root)
        if report != contract['expected_report']:
            raise ValueError('Medium report differs from the code-pinned version contract')
        for name, digest in contract['preserved_files'].items():
            if Path(name).is_absolute() or '..' in name.split('/') or '\\' in name:
                raise ValueError('Unsafe preserved-file path')
            if sha256(root / name) != digest:
                raise ValueError(f'Preserved Medium file changed: {name}')
        review = staging / 'medium_review'
        validate_review_bundle(review)
        inventory = lambda folder:{p.relative_to(folder).as_posix():sha256(p)
                                  for p in folder.rglob('*') if p.is_file()}
        if inventory(review) != inventory(root / 'review_bundle'):
            raise ValueError('Standalone Medium review differs from the candidate snapshot')
        output.mkdir()
        try:
            for path in sorted(staging.iterdir()):
                path.rename(output / path.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return dict(tag=contract['tag'], assets_verified=len(contract['assets']),
                extracted_to=str(output), dataset=report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_release(args.assets, args.manifest, args.output),
                     ensure_ascii=False, indent=2))
