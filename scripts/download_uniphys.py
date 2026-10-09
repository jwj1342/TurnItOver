#!/usr/bin/env python3
"""Download selected UniPhys-Bench Part 2 objects for TurnItOver.

Only fetches model.urdf, parts/, annotations/, and meta_data.json for the
requested object IDs. No full_model/ or point-cloud plys/ are downloaded.

Examples:
    python -m pip install -U huggingface_hub
    python scripts/download_uniphys.py --out-dir data/assets/uniphys
    python scripts/download_uniphys.py --ids UPB_00000000 --out-dir data/assets/uniphys

The downloader does NOT convert meshes or run Three.js / PR2. It is safe to
rerun: Hugging Face caches downloads, and successful files are verified again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
import xml.etree.ElementTree as ET

REPO_ID = "breezexian/UniPhys-Bench-Part2"
REPO_TYPE = "dataset"
DATA_PREFIX = "data"
MIN_ID, MAX_ID = 0, 453


def parse_ids(value: str) -> list[str]:
    """Accept a comma-separated list of numeric IDs or UPB_XXXXXXXX IDs."""
    ids: list[str] = []
    for item in value.split(","):
        raw = item.strip()
        if not raw:
            raise ValueError("Object IDs cannot contain empty entries")
        if raw.startswith("UPB_"):
            suffix = raw[4:]
            if len(suffix) != 8 or not suffix.isascii() or not suffix.isdecimal():
                raise ValueError(f"Invalid UniPhys ID: {raw}")
            number = int(suffix)
        elif raw.isascii() and raw.isdecimal():
            number = int(raw)
        else:
            raise ValueError(f"Invalid UniPhys ID: {raw}")
        if not MIN_ID <= number <= MAX_ID:
            raise ValueError(f"UniPhys-Bench Part 2 ID outside {MIN_ID}..{MAX_ID}: {raw}")
        name = f"UPB_{number:08d}"
        if name not in ids:
            ids.append(name)
    if not ids:
        raise ValueError("Provide at least one object ID")
    return ids


def selected_file(relative: str, object_id: str) -> bool:
    """Select the assembled URDF, part geometry/texture assets, and metadata."""
    p = PurePosixPath(relative)
    if (p.is_absolute() or ".." in p.parts or len(p.parts) < 3 or
            p.parts[:2] != (DATA_PREFIX, object_id)):
        return False
    local = p.parts[2:]
    if local in (("model.urdf",), ("meta_data.json",)):
        return True
    return len(local) >= 2 and local[0] in {"parts", "annotations"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry_network(operation, *, label: str, attempts: int = 4):
    """Retry transient HTTP transport failures without weakening TLS checks."""
    import httpx
    from huggingface_hub import close_session

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except httpx.TransportError as exc:
            if attempt == attempts:
                raise
            wait_seconds = 2 ** (attempt - 1)
            print(f"{label}: transient {type(exc).__name__}; retrying in "
                  f"{wait_seconds}s ({attempt}/{attempts - 1})", file=sys.stderr)
            close_session()
            time.sleep(wait_seconds)


def inspect_local_object(folder: Path) -> dict:
    """Check downloads and URDF mesh paths; does not claim Three.js compatibility."""
    urdf = folder / "model.urdf"
    if not urdf.is_file():
        raise FileNotFoundError(f"Missing URDF: {urdf}")
    root = ET.parse(urdf).getroot()
    if root.tag != "robot":
        raise ValueError(f"Invalid URDF root tag: {root.tag}")
    refs = set()
    unresolved, external = [], []
    for mesh in root.findall(".//visual/geometry/mesh"):
        name = mesh.get("filename", "")
        refs.add(name)
        rel = PurePosixPath(name)
        if not name or rel.is_absolute() or ".." in rel.parts or "://" in name:
            external.append(name)
            continue
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            unresolved.append(name)
    return {
        "urdf": str(urdf),
        "links": len(root.findall("link")),
        "joints": len(root.findall("joint")),
        "urdf_mesh_references": sorted(refs),
        "missing_local_meshes": sorted(set(unresolved)),
        "nonlocal_or_unsupported_mesh_uris": sorted(set(external)),
        "ready_for_urdf_import_precheck": not unresolved and not external,
    }


def fetch_object(api, hf_download, *, object_id: str, revision: str,
                 out_dir: Path, max_object_bytes: int, dry_run: bool) -> dict:
    """List a single upstream object directory before downloading any of it."""
    repo_path = f"{DATA_PREFIX}/{object_id}"
    rows = retry_network(
        lambda: list(api.list_repo_tree(REPO_ID, path_in_repo=repo_path,
                                        recursive=True, revision=revision,
                                        repo_type=REPO_TYPE)),
        label=f"[{object_id}] listing",
    )
    selected = sorted((r for r in rows if hasattr(r, "size") and
                       selected_file(r.path, object_id)), key=lambda r: r.path)
    names = [r.path for r in selected]
    if f"{repo_path}/model.urdf" not in names:
        raise ValueError(f"{repo_path}/model.urdf absent from the dataset file listing")
    if not any(p.startswith(f"{repo_path}/parts/") and p.lower().endswith(".obj") for p in names):
        raise ValueError(f"No OBJ part meshes listed in {repo_path}/parts/")
    unknown = [r.path for r in selected if r.size is None]
    if unknown:
        raise ValueError(f"Cannot enforce size limit: upstream size unavailable for {unknown[:3]}")
    total = sum(r.size for r in selected)
    if total > max_object_bytes:
        raise ValueError(f"Object size {total / 2**20:.1f} MiB exceeds --max-object-mb "
                         f"{max_object_bytes / 2**20:.1f}; raise the limit explicitly if intended")
    print(f"[{object_id}] {len(selected)} files, {total / 2**20:.1f} MiB", flush=True)
    if dry_run:
        return {"object_id": object_id, "status": "preview", "files": len(selected),
                "bytes": total, "paths": names}

    file_records = []
    for i, row in enumerate(selected, start=1):
        print(f"  [{i}/{len(selected)}] {row.path}", flush=True)
        path = Path(retry_network(
            lambda: hf_download(repo_id=REPO_ID, filename=row.path,
                                repo_type=REPO_TYPE, revision=revision,
                                local_dir=str(out_dir)),
            label=f"[{object_id}] {row.path}",
        ))
        if not path.is_file() or path.stat().st_size != row.size:
            raise IOError(f"Downloaded file missing or size mismatch: {row.path}")
        file_records.append({"path": row.path, "bytes": row.size, "sha256": sha256(path)})
    check = inspect_local_object(out_dir / repo_path)
    if check["missing_local_meshes"]:
        raise ValueError(f"URDF refers to missing meshes: {check['missing_local_meshes']}")
    return {"object_id": object_id, "status": "complete", "files": len(selected),
            "bytes": total, "records": file_records, "inspection": check}


def fetch_license_notice(api, hf_download, *, out_dir: Path, revision: str) -> dict | None:
    """Preserve the dataset-specific license notice, when published at the root."""
    entries = retry_network(
        lambda: list(api.list_repo_tree(REPO_ID, recursive=False, revision=revision,
                                        repo_type=REPO_TYPE)),
        label="License listing",
    )
    known = {"LICENSE_UNIPHYS_BENCH_PART2", "LICENSE", "LICENSE.md", "LICENSE.txt"}
    for row in entries:
        if getattr(row, "path", None) not in known or not hasattr(row, "size"):
            continue
        if row.size is None or row.size > 2 * 1024 * 1024:
            continue
        path = Path(retry_network(
            lambda: hf_download(repo_id=REPO_ID, filename=row.path,
                                repo_type=REPO_TYPE, revision=revision,
                                local_dir=str(out_dir)),
            label=f"License {row.path}",
        ))
        if path.stat().st_size != row.size:
            raise IOError(f"License notice size mismatch: {row.path}")
        return {"path": row.path, "sha256": sha256(path)}
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", default=",".join(str(i) for i in range(20)),
                        help="Comma-separated Part 2 IDs (default: 0..19)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/assets/uniphys"),
                        help="Local data root; relative paths are relative to current directory")
    parser.add_argument("--revision", default="main",
                        help="Hugging Face branch/tag/commit; always resolve to an immutable SHA")
    parser.add_argument("--max-object-mb", type=int, default=512,
                        help="Safety cap on listed bytes per object (default: 512 MiB)")
    parser.add_argument("--dry-run", action="store_true", help="List files and sizes only; do not download")
    args = parser.parse_args(argv)
    try:
        ids = parse_ids(args.ids)
        if args.max_object_mb < 1:
            raise ValueError("--max-object-mb must be positive")
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Install dependency: python -m pip install -U huggingface_hub") from exc

        api = HfApi()
        info = retry_network(
            lambda: api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE,
                                  revision=args.revision),
            label="Repository metadata",
        )
        revision = info.sha
        if not revision:
            raise RuntimeError("Could not resolve an immutable upstream revision")
        out_dir = args.out_dir.resolve()
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Source: {REPO_ID}@{revision}\nDestination: {out_dir}", flush=True)
        results = []
        for object_id in ids:
            try:
                result = fetch_object(api, hf_hub_download, object_id=object_id,
                                      revision=revision, out_dir=out_dir,
                                      max_object_bytes=args.max_object_mb * 1024 * 1024,
                                      dry_run=args.dry_run)
            except Exception as exc:
                result = {"object_id": object_id, "status": "failed",
                          "error_type": type(exc).__name__, "error": str(exc)}
                print(f"[{object_id}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            results.append(result)
        license_notice = None
        if not args.dry_run:
            try:
                license_notice = fetch_license_notice(api, hf_hub_download,
                                                       out_dir=out_dir, revision=revision)
            except Exception as exc:
                print(f"Warning: could not download dataset license notice: {exc}", file=sys.stderr)
        summary = {"dataset": REPO_ID, "repo_type": REPO_TYPE, "revision": revision,
                   "requested_revision": args.revision, "license": "CC-BY-NC-4.0",
                   "license_notice": license_notice,
                   "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                   "out_dir": str(out_dir), "mode": "dry-run" if args.dry_run else "download",
                   "results": results,
                   "counts": {status: sum(r["status"] == status for r in results)
                              for status in ("complete", "preview", "failed")}}
        if not args.dry_run:
            manifest = out_dir / "download-manifest.json"
            manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"\nManifest: {manifest}")
        print("\n" + json.dumps(summary["counts"], ensure_ascii=False))
        return 0 if not any(r["status"] == "failed" for r in results) else 1
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
