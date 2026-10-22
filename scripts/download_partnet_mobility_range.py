#!/usr/bin/env python3
"""Download the next slice of the seeded PartNet-Mobility order: archives [skip, skip + count).

Wraps ``download_partnet_mobility`` without changing it, because that file's SHA is pinned by released
evidence manifests. The seeded order is identical, so a slice never repeats an earlier download.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import download_partnet_mobility as base
from turnitover.core.jsonio import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", default="ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--skip", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--max-archive-mb", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    original = base.select_files
    base.select_files = lambda siblings, *, count, seed, max_bytes: original(
        siblings, count=count + args.skip, seed=seed, max_bytes=max_bytes)[args.skip:]
    path = base.download(args.out.resolve(), revision=args.revision, count=args.count, seed=args.seed,
                         max_archive_mb=args.max_archive_mb, workers=args.workers)
    manifest = json.loads(path.read_text())
    manifest["selection"]["skip"] = args.skip
    write_json_atomic(path, manifest)
    print(path)


if __name__ == "__main__":
    main()
