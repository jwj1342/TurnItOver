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
    with pytest.raises(ValueError, match="different inputs"):
        run_generate(dataclasses.replace(cfg, seed=cfg.seed + 1), 0, 1)


def test_catalog_clean_and_mixed_are_shard_independent(tmp_path, toy_spec):
    from turnitover.core.serde import to_dict
    entries = []
    for i in range(2):
        spec = dataclasses.replace(toy_spec, asset_id=f"fixture-{i}")
        (tmp_path / f"spec-{i}.json").write_text(json.dumps(to_dict(spec)))
        entries.append({"spec": f"spec-{i}.json", "split_group": "same-toy-family", "origin": "test fixture", "license": "fixture"})
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"version": 1, "source_id": "test", "assets": entries}))
    cfg = load_generate_config(ROOT / "configs/generate_dataset.yaml", ROOT)
    cfg = dataclasses.replace(cfg, n_samples=4, seed=1, asset_source={"kind": "catalog", "manifest": str(catalog)},
                              output_dir=tmp_path / "whole", clean_fraction=0.5,
                              render=dataclasses.replace(cfg.render, width=128, height=128))
    whole = [s for s, _ in iter_samples(run_generate(cfg))]
    split_cfg = dataclasses.replace(cfg, output_dir=tmp_path / "split")
    pieces = [s for shard in range(2) for s, _ in iter_samples(run_generate(split_cfg, shard, 2))]
    def stable(samples):
        return sorted((s.idx, s.asset_id, s.program_sha, s.corruptions, s.dataset["quality"]) for s in samples)
    assert stable(whole) == stable(pieces)
    assert {s.asset_id for s in whole} == {"fixture-0", "fixture-1"}
    assert any(not s.corruptions for s in whole)
    assert any(len(s.corruptions) == 2 for s in whole)
    assert all(s.dataset["quality"]["status"] == "accepted" for s in whole if not s.corruptions)
