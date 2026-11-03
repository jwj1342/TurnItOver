"""Canonical inventory for portable Hard candidate and review bundles."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

if __package__:
    from .easy_release_archive import sha256
else:
    from easy_release_archive import sha256

NAME = "bundle_manifest.json"


def _files(root: Path):
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Hard bundle must not contain symbolic links")
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != NAME
    }


def write_manifest(root: Path):
    files = {name: sha256(path) for name, path in sorted(_files(root).items())}
    (root / NAME).write_text(
        json.dumps({"version": 1, "files": files}, indent=2, sort_keys=True) + "\n"
    )
    return files


def validate_manifest(root: Path):
    encoded = (root / NAME).read_text()
    manifest = json.loads(encoded)
    canonical = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if (
        set(manifest) != {"version", "files"}
        or manifest.get("version") != 1
        or not isinstance(manifest.get("files"), dict)
        or encoded != canonical
    ):
        raise ValueError("Invalid or noncanonical Hard bundle inventory")
    actual = _files(root)
    if set(actual) != set(manifest["files"]):
        raise ValueError("Hard inventory has missing or unlisted files")
    for name, digest in manifest["files"].items():
        if (
            "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
            or sha256(actual[name]) != digest
        ):
            raise ValueError(f"Invalid path or SHA-256 mismatch: {name}")
    return set(actual)
