import dataclasses
import json
from pathlib import Path

import pytest

from turnitover.config import load_generate_config
from turnitover.engine.generate import run_generate
from turnitover.storage.reader import iter_samples, read_index

pytestmark = pytest.mark.browser
ROOT = Path(__file__).resolve().parents[2]


def test_generate_three_samples(tmp_path):
    cfg = load_generate_config(ROOT / "configs/generate_toy.yaml", ROOT)
    cfg = dataclasses.replace(cfg, n_samples=3, output_dir=tmp_path, render=dataclasses.replace(cfg.render, width=192, height=192))
    tar = run_generate(cfg, 0, 1)
    assert tar.exists()
    rows = read_index(tmp_path / "shard-0000.index.jsonl")
    assert len(rows) == 3
    manifest = json.loads((tmp_path / "shard-0000.manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["n_ok"] == 3
    samples = list(iter_samples(tar))
    assert len(samples) == 3
    for s, members in samples:
        assert s.compile_ok and len(s.trajectory) == 7
        assert sum(m.startswith("obs/") for m in members) == 4
        assert "geometry.npz" in members
        assert {c.checker_id for c in s.checks} == {"geometry.alignment", "runtime.budget"}
    failed = [c for s, _ in samples for c in s.checks if not c.passed]
    assert failed, "every corruption should trip at least one checker"
    # idempotent re-run
    assert run_generate(cfg, 0, 1) == tar
