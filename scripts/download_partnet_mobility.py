#!/usr/bin/env python3
"""Download a deterministic official PartNet-Mobility pilot from Hugging Face."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from turnitover.core.jsonio import write_json_atomic

REPO_ID = "sapien-sim/PartNetMobility"
REPO_TYPE = "dataset"
LICENSE = "PartNet-Mobility terms: non-commercial research/education; ShapeNet terms also apply"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_files(siblings, *, count: int, seed: int, max_bytes: int, skip: int = 0) -> list:
    eligible = [row for row in siblings
                if re.fullmatch(r"\d+\.zip", row.rfilename) and row.size is not None
                and row.size <= max_bytes]
    if count < 1 or skip < 0 or skip + count > len(eligible):
        raise ValueError(f"--skip + --count must be in 1..{len(eligible)}")
    ranked = sorted(eligible, key=lambda row: hashlib.sha256(f"{seed}:{row.rfilename}".encode()).digest())
    return ranked[skip:skip + count]


def download(output: Path, *, revision: str, count: int, seed: int,
             max_archive_mb: int, workers: int = 8, skip: int = 0) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(REPO_ID, revision=revision, files_metadata=True)
    selected = select_files(info.siblings, count=count, seed=seed,
                            max_bytes=max_archive_mb * 1024 * 1024, skip=skip)
    output.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        path = Path(hf_hub_download(REPO_ID, row.rfilename, repo_type=REPO_TYPE,
                                    revision=info.sha, local_dir=output))
        if path.stat().st_size != row.size:
            raise IOError(f"Downloaded size mismatch: {row.rfilename}")
        return {"asset_id": path.stem, "path": row.rfilename, "bytes": row.size,
                "sha256": _sha256(path)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(fetch, selected))
    manifest = {
        "version": 1, "status": "complete", "dataset": REPO_ID,
        "dataset_revision": info.sha, "license": LICENSE,
        "license_urls": [
            "https://huggingface.co/datasets/sapien-sim/PartNetMobility",
            "https://huggingface.co/datasets/ShapeNet/ShapeNetCore",
        ],
        "selection": {"method": "sha256(seed:filename)", "seed": seed,
                      "count": count, "skip": skip, "max_archive_mb": max_archive_mb},
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archives": rows,
    }
    path = output / "download-manifest.json"
    write_json_atomic(path, manifest)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/partnet-mobility-pilot"))
    parser.add_argument("--revision", default="ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f")
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--max-archive-mb", type=int, default=5)
    parser.add_argument("--skip", type=int, default=0, help="skip the first N archives of the seeded order")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(download(args.out.resolve(), revision=args.revision, count=args.count,
                   seed=args.seed, max_archive_mb=args.max_archive_mb,
                   workers=args.workers, skip=args.skip))


if __name__ == "__main__":
    main()
