"""ShardWriter: append samples to a tar with O(1) files per shard."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np

from turnitover.checkers.base import PartMesh
from turnitover.core.sample import Sample
from turnitover.core.serde import to_dict


def shard_paths(out_dir: Path, shard: int) -> dict[str, Path]:
    stem = out_dir / f"shard-{shard:04d}"
    return {"tar": stem.with_suffix(".tar"), "index": Path(f"{stem}.index.jsonl"), "manifest": Path(f"{stem}.manifest.json")}


def index_line(sample: Sample) -> dict:
    return {
        "sample_id": sample.sample_id,
        "idx": sample.idx,
        "seed": sample.seed,
        "asset_id": sample.asset_id,
        "defect_ids": [c.defect_id for c in sample.corruptions],
        "parts": sorted({p for c in sample.corruptions for p in c.parts}),
        "severity": max((c.severity for c in sample.corruptions), default=0.0),
        "compile_ok": sample.compile_ok,
        "checks_passed": {c.checker_id: c.passed for c in sample.checks},
    }


class ShardWriter:
    def __init__(self, out_dir: Path, shard: int):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.paths = shard_paths(out_dir, shard)
        self._tar = tarfile.open(self.paths["tar"], "w")
        self._index = self.paths["index"].open("w", encoding="utf-8")
        self.count = 0

    def add(self, sample: Sample, images: dict[str, bytes], geometry: dict[str, dict[str, PartMesh]]) -> None:
        prefix = sample.sample_id
        self._add_bytes(f"{prefix}/record.json", json.dumps(to_dict(sample), ensure_ascii=False).encode("utf-8"))
        for ref, png in images.items():
            self._add_bytes(f"{prefix}/{ref}", png)
        if geometry:
            self._add_bytes(f"{prefix}/geometry.npz", _npz_bytes(geometry))
        self._index.write(json.dumps(index_line(sample), ensure_ascii=False) + "\n")
        self.count += 1

    def write_manifest(self, manifest: dict) -> None:
        self.paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def close(self) -> None:
        self._tar.close()
        self._index.close()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _add_bytes(self, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        self._tar.addfile(info, io.BytesIO(data))


def _npz_bytes(geometry: dict[str, dict[str, PartMesh]]) -> bytes:
    arrays = {}
    for state, parts in geometry.items():
        for pid, mesh in parts.items():
            arrays[f"{state}/{pid}/vertices"] = mesh.vertices
            arrays[f"{state}/{pid}/faces"] = mesh.faces
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()
