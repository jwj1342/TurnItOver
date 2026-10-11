import dataclasses
import json
from types import SimpleNamespace

from scripts import convert_uniphys, prepare_uniphys_verifier_benchmark, select_uniphys_trainval
from turnitover.core.serde import to_dict


def _write_asset(root, object_id, category="fixture"):
    folder = root / "data" / object_id
    folder.mkdir(parents=True)
    (folder / "model.urdf").write_text("""<robot name="fixture">
      <link name="base"/><link name="door"/>
      <joint name="hinge" type="revolute"><parent link="base"/><child link="door"/>
        <limit lower="0" upper="1"/></joint>
    </robot>""")
    (folder / "meta_data.json").write_text(json.dumps({"object": {"category": category}}))


def test_selects_disjoint_deterministic_train_validation_objects(tmp_path):
    root = tmp_path / "source"
    revision = "c" * 40
    root.mkdir()
    (root / "download-manifest.json").write_text(json.dumps({
        "dataset": "breezexian/UniPhys-Bench-Part2", "revision": revision,
        "license": "CC-BY-NC-4.0",
    }))
    for index in range(5):
        _write_asset(root, f"UPB_{index:08d}", category=f"category-{index % 2}")
    test_group = f"uniphys:{revision}:UPB_00000000"
    test_splits = tmp_path / "test-splits.json"
    test_splits.write_text(json.dumps({"version": 1, "unit": "split_group",
                                       "groups": {test_group: "test"}}))

    first = select_uniphys_trainval.select(
        root, test_splits, tmp_path / "first.json", train_objects=2,
        validation_objects=1, seed=7)
    second = select_uniphys_trainval.select(
        root, test_splits, tmp_path / "second.json", train_objects=2,
        validation_objects=1, seed=7)

    assert first == second
    assert first["expected_object_counts"] == {"train": 2, "validation": 1}
    assert "UPB_00000000" not in {row["id"] for row in first["objects"]}
    assert {row["split"] for row in first["objects"]} == {"train", "validation"}
    ids, assignments, _ = convert_uniphys._load_split_plan(tmp_path / "first.json")
    assert len(ids) == 3 and set(assignments.values()) == {"train", "validation"}


def test_builds_three_cases_per_object_into_separate_gold_files(tmp_path, toy_spec, monkeypatch):
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir()
    assets = []
    assignments = {}
    for index, split in enumerate(("train", "validation")):
        object_id = f"UPB_{index + 30:08d}"
        spec = dataclasses.replace(toy_spec, asset_id=object_id, category=f"category-{index}")
        (catalog_root / f"{object_id}.json").write_text(json.dumps(to_dict(spec)))
        group = f"uniphys:test:{object_id}"
        assets.append({"spec": f"{object_id}.json", "split_group": group,
                       "origin": "fixture", "license": "fixture"})
        assignments[group] = split
    catalog = catalog_root / "catalog.json"
    catalog.write_text(json.dumps({
        "version": 1, "source_id": "uniphys-trainval", "dataset_revision": "d" * 40,
        "evaluation_only": False, "role": "train_validation", "assets": assets,
    }))
    splits = catalog_root / "trainval-splits.json"
    splits.write_text(json.dumps({
        "version": 1, "unit": "split_group", "groups": assignments,
        "expected_object_counts": {"train": 1, "validation": 1},
    }))

    class FakeSession:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def load(self, *args, **kwargs): pass
        def request_view(self, *args): return SimpleNamespace(png=b"png")

    def fake_validate(candidate, reference, *args):
        return {"status": "complete", "code_correct": candidate.sha == reference.sha}

    monkeypatch.setattr(prepare_uniphys_verifier_benchmark, "ObservationSession", FakeSession)
    monkeypatch.setattr(prepare_uniphys_verifier_benchmark, "validate_code", fake_validate)
    output = tmp_path / "benchmark"
    report = prepare_uniphys_verifier_benchmark.run(
        catalog, output, splits_path=splits, seed=11)

    assert report["object_counts"] == {"train": 1, "validation": 1}
    assert report["case_counts"] == {"train": 3, "validation": 3}
    assert len((output / "train.gold.jsonl").read_text().splitlines()) == 3
    assert len((output / "validation.gold.jsonl").read_text().splitlines()) == 3
    assert not (output / "test.gold.jsonl").exists()
