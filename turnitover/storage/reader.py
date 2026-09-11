"""Read shard tars back into Samples (plus raw member bytes)."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Iterator

import numpy as np

from turnitover.checkers.base import PartMesh
from turnitover.core.sample import Sample
from turnitover.core.serde import from_dict


def read_index(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_samples(tar_path: Path) -> Iterator[tuple[Sample, dict[str, bytes]]]:
    """Yield (sample, members) where members maps 'obs/00.png' / 'geometry.npz' -> bytes."""
    with tarfile.open(tar_path, "r") as tar:
        current: str | None = None
        members: dict[str, bytes] = {}
        record: Sample | None = None
        for m in tar:
            sid, _, rel = m.name.partition("/")
            if sid != current:
                if record is not None:
                    yield record, members
                current, members, record = sid, {}, None
            data = tar.extractfile(m).read()
            if rel == "record.json":
                record = from_dict(Sample, json.loads(data))
            else:
                members[rel] = data
        if record is not None:
            yield record, members


def load_geometry(npz_bytes: bytes) -> dict[str, dict[str, PartMesh]]:
    out: dict[str, dict[str, PartMesh]] = {}
    with np.load(io.BytesIO(npz_bytes)) as z:
        keys = {k.rsplit("/", 1)[0] for k in z.files}
        for key in keys:
            state, pid = key.split("/", 1)
            out.setdefault(state, {})[pid] = PartMesh(vertices=z[f"{key}/vertices"], faces=z[f"{key}/faces"])
    return out
