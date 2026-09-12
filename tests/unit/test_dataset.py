import dataclasses
import json

import pytest

from tests.unit.test_serde import _sample
from turnitover.assets.catalog import CatalogSource, decode_spec, validate_spec
from turnitover.core.sample import CheckResult, CorruptionSpec, Invariant
from turnitover.core.serde import from_dict, to_dict
from turnitover.core.sample import Sample
from turnitover.dataset.prepare import make_splits, prepare
from turnitover.dataset.evaluate import evaluate_runtime
from turnitover.engine.quality import qualify, validate_label
from turnitover.storage.writer import ShardWriter


def test_triangle_perturbation_is_not_automatically_a_violation():
    label = CorruptionSpec("runtime.triangle_budget", ("door",), 0.5)
    under = (CheckResult("runtime.budget", True, (Invariant("triangles", 100, 5000, True),)),)
    over = (CheckResult("runtime.budget", False, (Invariant("triangles", 6000, 5000, False),)),)
    assert validate_label(label, under)["status"] == "invalid"
    assert validate_label(label, over)["status"] == "valid"
    assert qualify((label,), over, (under,))["status"] == "quarantined"
    assert qualify((), under, ())["status"] == "accepted"
    assert qualify((), (), ())["status"] == "quarantined"


def test_kinematic_label_requires_target_part_and_state():
    label = CorruptionSpec("kinematics.joint_axis", ("door",), 0.5, {"joint": "door"})
    wrong_state = (CheckResult("geometry.alignment", False, (Invariant("chamfer_m", 1, .01, False, ("door",), "rest"),)),)
    assert validate_label(label, wrong_state)["status"] == "unsupported"


def test_catalog_rejects_bad_hierarchy_and_keeps_variant_group(tmp_path, toy_spec):
    broken = toy_spec.replace_part("door", parent="door")
    with pytest.raises(ValueError, match="cycle"):
        validate_spec(broken)
    entries = []
    for name in ("a", "b"):
        spec = dataclasses.replace(toy_spec, asset_id=name)
        (tmp_path / f"{name}.json").write_text(json.dumps(to_dict(spec)))
        entries.append({"spec": f"{name}.json", "split_group": "same-original", "origin": "unit test", "license": "fixture"})
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"version": 1, "source_id": "fixtures", "assets": entries}))
    source = CatalogSource(str(path))
    assert [s.asset_id for s in source] == ["a", "b"]
    assert source.metadata("a")["split_group"] == source.metadata("b")["split_group"]


def test_catalog_never_silently_discards_meshes_or_vector_entries(toy_spec):
    raw = to_dict(toy_spec)
    raw["parts"][0]["mesh"] = "real.glb"
    with pytest.raises(ValueError, match="Unsupported fields"):
        decode_spec(raw)
    del raw["parts"][0]["mesh"]
    raw["parts"][0]["origin"].append(4)
    with pytest.raises(ValueError, match="vector length"):
        decode_spec(raw)


def _write_shard(tmp_path, *, leak=False, bad_quality=False):
    with ShardWriter(tmp_path, 0) as writer:
        for i in range(3):
            s = _sample()
            s = dataclasses.replace(s, sample_id=f"fixture-{i}", asset_id=f"asset-{i}", reference_sha="same" if leak else f"ref-{i}",
                corruptions=(), dataset={"category": "fixture", "split_group": f"group-{i}", "origin": "unit test", "license": "fixture",
                    "quality": {"status": "quarantined" if bad_quality and i == 0 else "accepted"},
                    "reference_images": ["reference/00.png"], "parts": ["door"], "joints": ["door"], "views": ["front"],
                    "constraints": {}, "reference_program_source": "SECRET_REFERENCE"})
            writer.add(s, {"reference/00.png": b"ref", "obs/00.png": b"obs", "obs/01.png": b"obs"}, {})
        writer.write_manifest({"status": "complete"})
    return [tmp_path / "shard-0000.tar"]


def test_splits_export_privilege_boundary_quarantine_and_evaluation(tmp_path):
    shards = _write_shard(tmp_path / "raw", bad_quality=True)
    split_path = tmp_path / "splits.json"
    mapping = make_splits(shards, split_path, seed=42)
    assert set(mapping.values()) == {"train", "validation", "test"}
    output = tmp_path / "prepared"
    result = prepare(shards, split_path, output)
    assert sum(result["counts"].values()) == 2
    assert result["quality_counts"]["quarantined"] == 1
    public = "".join(p.read_text() for p in output.glob("*.inputs.jsonl"))
    for forbidden in ("SECRET_REFERENCE", "program_source", "corruptions", "split_group", "severity"):
        assert forbidden not in public
    split = mapping["group-1"]
    metrics = evaluate_runtime(output, split, tmp_path / "metrics")
    assert metrics["n"] == 1 and metrics["uncertain_rate"] == 1
    with pytest.raises(ValueError, match="must not exist"):
        prepare(shards, split_path, output)


def test_reference_leakage_and_legacy_samples_rejected(tmp_path):
    shards = _write_shard(tmp_path / "raw", leak=True)
    splits = tmp_path / "splits.json"
    make_splits(shards, splits)
    with pytest.raises(ValueError, match="leakage"):
        prepare(shards, splits, tmp_path / "bad")
    assert json.loads((tmp_path / "bad/manifest.json").read_text())["status"] == "incomplete"
    old = to_dict(_sample())
    old.pop("dataset")
    old["schema_version"] = 1
    assert from_dict(Sample, old).dataset == {}


def test_single_asset_cannot_be_randomly_split(tmp_path):
    with ShardWriter(tmp_path / "raw", 0) as writer:
        writer.add(dataclasses.replace(_sample(), dataset={"split_group": "one"}), {}, {})
        writer.write_manifest({"status": "complete"})
    with pytest.raises(ValueError, match="three independent"):
        make_splits([tmp_path / "raw/shard-0000.tar"], tmp_path / "splits.json")


def test_evaluation_counts_uncertain_as_miss_with_undefined_precision(tmp_path):
    shards = _write_shard(tmp_path / "raw")
    splits = tmp_path / "splits.json"
    make_splits(shards, splits)
    prepared = tmp_path / "prepared"
    prepare(shards, splits, prepared)
    gold_path = prepared / "test.gold.jsonl"
    row = json.loads(gold_path.read_text())
    row["target"]["findings"] = [{"defect_id": "kinematics.joint_axis"}]
    gold_path.write_text(json.dumps(row) + "\n")
    metrics = evaluate_runtime(prepared, "test", tmp_path / "eval")
    assert metrics["uncertain_rate"] == 1
    assert metrics["per_class"]["kinematics.joint_axis"] == {
        "tp": 0, "fp": 0, "fn": 1, "precision": None, "recall": 0, "f1": 0}
