"""Verify pinned release assets and fully validate them before publishing extraction."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

if __package__:
    from .easy_release_archive import extract_zip, sha256, verify_asset_directory
    from .prepare_easy_validation_review import validate_review_bundle
    from .validate_easy_bundle import validate_bundle
else:
    from easy_release_archive import extract_zip, sha256, verify_asset_directory
    from prepare_easy_validation_review import validate_review_bundle
    from validate_easy_bundle import validate_bundle


def verify_release_assets(assets: Path, manifest: Path) -> dict:
    return verify_asset_directory(assets, manifest, {"easy", "validation_review"})


def validate_release(assets: Path, manifest: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("extraction output must not exist")
    contract = verify_release_assets(assets, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".easy-release-", dir=output.parent) as temporary:
        staging = Path(temporary) / "validated"
        staging.mkdir()
        for name, expected in contract["assets"].items():
            extract_zip(assets / name, staging, expected)
        root = staging / "easy"
        report = validate_bundle(root)
        if report != contract["expected_report"]:
            raise ValueError("full dataset report differs from the version contract")
        for relative, expected_hash in contract["preserved_files"].items():
            if relative.startswith("/") or ".." in relative.split("/") or "\\" in relative:
                raise ValueError("unsafe preserved file path")
            if sha256(root / relative) != expected_hash:
                raise ValueError(f"preserved dataset file changed: {relative}")
        review = staging / "validation_review"
        validate_review_bundle(review)
        inventory = lambda directory: {path.relative_to(directory).as_posix(): sha256(path)
                                       for path in directory.rglob("*") if path.is_file()}
        if inventory(review) != inventory(root / "validation_review"):
            raise ValueError("standalone review differs from the data snapshot")
        output.mkdir()
        try:
            for directory in sorted(staging.iterdir()):
                directory.rename(output / directory.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return dict(tag=contract["tag"], assets_verified=len(contract["assets"]),
                extracted_to=str(output), dataset=report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="contract from the pinned code checkout")
    parser.add_argument("--output", type=Path, required=True, help="new directory, created only after complete validation")
    args = parser.parse_args()
    print(json.dumps(validate_release(args.assets, args.manifest, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
