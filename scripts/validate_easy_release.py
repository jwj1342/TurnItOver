"""Verify pinned release assets and fully validate them before publishing extraction."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

if __package__:
    from .easy_release_archive import extract_zip, sha256, verify_asset
    from .prepare_easy_validation_review import validate_review_bundle
    from .validate_easy_bundle import validate_bundle
else:
    from easy_release_archive import extract_zip, sha256, verify_asset
    from prepare_easy_validation_review import validate_review_bundle
    from validate_easy_bundle import validate_bundle


def verify_release_assets(assets: Path, manifest: Path) -> dict:
    contract = json.loads(manifest.read_text(encoding="utf-8"))
    if contract.get("version") != 1 or len(contract.get("assets", {})) != 2:
        raise ValueError("unsupported Easy release contract")
    expected_roots = {"easy", "validation_review"}
    if {entry["root"] for entry in contract["assets"].values()} != expected_roots:
        raise ValueError("release must contain data and independent review assets")
    for name, entry in contract["assets"].items():
        if Path(name).name != name or "\\" in name:
            raise ValueError("release asset name must be a filename")
        verify_asset(assets / name, entry)
    copied = assets / "release-manifest.json"
    if copied.is_symlink() or copied.read_bytes() != manifest.read_bytes():
        raise ValueError("downloaded manifest differs from the code-pinned contract")
    names = sorted([*contract["assets"], copied.name])
    checksums = "".join(f"{sha256(assets / name)}  {name}\n" for name in names)
    if (assets / "SHA256SUMS").read_text(encoding="utf-8") != checksums:
        raise ValueError("SHA256SUMS differs from the pinned assets")
    if {path.name for path in assets.iterdir()} != {*names, "SHA256SUMS"}:
        raise ValueError("unexpected or missing files in release asset directory")
    return contract


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
