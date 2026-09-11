"""Stateless partitioning of the sample index space and stable per-sample seeds."""
from __future__ import annotations

import numpy as np


def shard_indices(n_samples: int, shard: int, n_shards: int) -> range:
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} out of range for {n_shards} shards")
    return range(shard, n_samples, n_shards)


def sample_seed(base_seed: int, idx: int) -> int:
    return int(np.random.SeedSequence([base_seed, idx]).generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def parse_shard(text: str) -> tuple[int, int]:
    i, _, n = text.partition("/")
    return int(i), int(n or 1)
