"""Deterministic ZIP assets and constrained extraction for Easy releases."""

from __future__ import annotations

import hashlib
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_zip(root: Path, destination: Path, prefix: str) -> dict:
    if destination.exists() or destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("ZIP destination must be new and outside its source")
    paths = sorted(root.rglob("*"))
    if root.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("release source must not contain symbolic links")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count, size = 0, 0
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(f"{prefix}/{path.relative_to(root).as_posix()}", (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as source, archive.open(info, "w") as target:
                shutil.copyfileobj(source, target)
            count += 1
            size += path.stat().st_size
    return dict(sha256=sha256(destination), size=destination.stat().st_size,
                root=prefix, files=count, uncompressed_bytes=size)


def verify_asset(path: Path, expected: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symbolic release asset: {path.name}")
    if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"release asset size or SHA-256 mismatch: {path.name}")


def extract_zip(path: Path, destination: Path, expected: dict) -> Path:
    """Verify first; accept only regular files below one version-pinned root."""
    verify_asset(path, expected)
    root = destination / expected["root"]
    if root.exists() or root.is_symlink():
        raise ValueError("extraction root must not exist")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or len(names) != expected["files"]:
            raise ValueError("ZIP file inventory is duplicate or incomplete")
        if sum(entry.file_size for entry in entries) != expected["uncompressed_bytes"]:
            raise ValueError("ZIP uncompressed size differs from the version contract")
        for entry in entries:
            name = entry.filename
            parts = PurePosixPath(name).parts
            mode = entry.external_attr >> 16
            if ("\\" in name or not parts or parts[0] != expected["root"]
                    or len(parts) < 2 or any(part in {"", ".", ".."} for part in name.split("/"))
                    or entry.is_dir() or stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
                raise ValueError(f"unsafe ZIP member: {name!r}")
        for entry in entries:
            target = destination.joinpath(*PurePosixPath(entry.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    return root
