#!/usr/bin/env python3
"""Run the existing three-view browser motion gate over selected catalog assets."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.render_uniphys40k_smoke import run
from turnitover.assets.catalog import CatalogSource
from turnitover.core.jsonio import write_json_atomic


def audit(catalog: Path, output: Path, ids: list[str], *, size: int = 512) -> Path:
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    source = CatalogSource(str(catalog))
    known = {spec.asset_id for spec in source}
    if not ids or len(set(ids)) != len(ids) or set(ids) - known:
        raise ValueError("Selected IDs must be unique and present in the catalog")
    output.mkdir(parents=True)
    rows = []
    for index, asset_id in enumerate(ids, start=1):
        try:
            smoke = run(catalog, output / asset_id, asset_id, size=size, source=source)
            row = {"asset_id": asset_id, "status": "mechanical_checks_complete",
                   "smoke_json": str(smoke.resolve())}
        except Exception as exc:
            row = {"asset_id": asset_id, "status": "mechanical_check_failed",
                   "error_type": type(exc).__name__, "reason": str(exc)}
        rows.append(row)
        print(json.dumps({"index": index, "total": len(ids), **row}), flush=True)
    counts = Counter(row["status"] for row in rows)
    path = output / "audit.json"
    write_json_atomic(path, {
        "version": 1, "status": "complete", "catalog": str(catalog.resolve()),
        "render_size": size, "physical_review": "required",
        "assets": rows, "counts": dict(counts),
    })
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids", required=True, help="Comma-separated AssetSpec IDs")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    print(audit(args.catalog.resolve(), args.out.resolve(),
                [value.strip() for value in args.ids.split(",") if value.strip()],
                size=args.size))


if __name__ == "__main__":
    main()
