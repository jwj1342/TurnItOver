"""Verify downloaded Hard assets before publishing their extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

if __package__:
    from .easy_release_archive import extract_zip, sha256, verify_asset_directory
    from .prepare_hard_review import validate_review_bundle
    from .validate_hard_bundle import validate_bundle
else:
    from easy_release_archive import extract_zip, sha256, verify_asset_directory
    from prepare_hard_review import validate_review_bundle
    from validate_hard_bundle import validate_bundle


def validate_release(assets: Path, manifest: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("Extraction output must not exist")
    contract = verify_asset_directory(assets, manifest, {"hard", "hard_review"})
    if contract.get("dataset") != "hard" or contract.get("training_ready") is not False:
        raise ValueError("Expected non-training Hard candidate contract")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".hard-release-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "validated"
        staging.mkdir()
        for name, entry in contract["assets"].items():
            extract_zip(assets / name, staging, entry)
        root = staging / "hard"
        report = validate_bundle(root)
        if report != contract["expected_report"]:
            raise ValueError("Hard report differs from pinned contract")
        for name, digest in contract["preserved_files"].items():
            if Path(name).is_absolute() or ".." in name.split("/") or "\\" in name:
                raise ValueError("Unsafe preserved path")
            if sha256(root / name) != digest:
                raise ValueError(f"Preserved Hard file changed: {name}")
        review = staging / "hard_review"
        validate_review_bundle(review)
        inventory = lambda folder: {
            p.relative_to(folder).as_posix(): sha256(p)
            for p in folder.rglob("*")
            if p.is_file()
        }
        if inventory(review) != inventory(root / "review_bundle"):
            raise ValueError("Standalone Hard review differs")
        output.mkdir()
        try:
            for path in sorted(staging.iterdir()):
                path.rename(output / path.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return {
        "tag": contract["tag"],
        "assets_verified": len(contract["assets"]),
        "extracted_to": str(output),
        "dataset": report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(validate_release(args.assets, args.manifest, args.output), indent=2)
    )
