import dataclasses
import hashlib
import json

import pytest

from scripts import convert_uniphys
from scripts.prepare_uniphys_verifier_benchmark import _load_frozen_catalog
from turnitover.assets.catalog import CatalogSource
from turnitover.core.serde import to_dict


def test_convert_uniphys_freezes_provenance_and_evaluation_split_group(tmp_path, toy_spec, monkeypatch):
    root = tmp_path / "source"
    folder = root / "data/UPB_00000000"
    folder.mkdir(parents=True)
    urdf = folder / "model.urdf"
    metadata = folder / "meta_data.json"
    urdf.write_text("fixture urdf")
    metadata.write_text(json.dumps({"object": {"category": "fixture"}}))
    revision = "a" * 40
    (root / "download-manifest.json").write_text(json.dumps({
        "dataset": "breezexian/UniPhys-Bench-Part2", "revision": revision,
        "license": "CC-BY-NC-4.0", "results": [],
    }))

    class FakeUrdfModel:
        continuous_joints_fixed_at_zero = ("spin",)

        def __init__(self, path, *, continuous_joint_policy):
            assert path == urdf and continuous_joint_policy == "fixed"

        def spec(self, asset_id, category):
            return dataclasses.replace(toy_spec, asset_id=asset_id, category=category)

        def provenance(self):
            return [{"path": str(urdf.resolve()),
                     "sha256": hashlib.sha256(urdf.read_bytes()).hexdigest()}]

    monkeypatch.setattr(convert_uniphys, "UrdfModel", FakeUrdfModel)
    output = tmp_path / "catalog"
    catalog_path = convert_uniphys.convert(root, output, ("UPB_00000000",))

    source = CatalogSource(str(catalog_path))
    assert source.source_id == f"uniphys-{revision}"
    assert source.metadata("UPB_00000000")["split_group"] == (
        f"uniphys:{revision}:UPB_00000000")
    splits = json.loads((output / "test-splits.json").read_text())
    assert splits["groups"] == {f"uniphys:{revision}:UPB_00000000": "test"}
    report = json.loads((output / "conversion.json").read_text())
    assert report["status"] == "complete" and report["role"] == "evaluation_only"
    assert report["assets"][0]["asset_sha256"]
    assert report["assets"][0]["continuous_joints_fixed_at_zero"] == ["spin"]


def test_convert_uniphys_writes_explicit_train_validation_splits(tmp_path, toy_spec, monkeypatch):
    root = tmp_path / "source"
    object_ids = ("UPB_00000030", "UPB_00000031")
    revision = "b" * 40
    (root / "data").mkdir(parents=True)
    (root / "download-manifest.json").write_text(json.dumps({
        "dataset": "breezexian/UniPhys-Bench-Part2", "revision": revision,
        "license": "CC-BY-NC-4.0", "results": [],
    }))
    for object_id in object_ids:
        folder = root / "data" / object_id
        folder.mkdir()
        (folder / "model.urdf").write_text("fixture urdf")
        (folder / "meta_data.json").write_text(json.dumps({"object": {"category": "fixture"}}))

    class FakeUrdfModel:
        continuous_joints_fixed_at_zero = ()

        def __init__(self, path, *, continuous_joint_policy):
            self.path = path

        def spec(self, asset_id, category):
            return dataclasses.replace(toy_spec, asset_id=asset_id, category=category)

        def provenance(self):
            return [{"path": str(self.path.resolve()),
                     "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}]

    monkeypatch.setattr(convert_uniphys, "UrdfModel", FakeUrdfModel)
    assignments = {object_ids[0]: "train", object_ids[1]: "validation"}
    catalog_path = convert_uniphys.convert(
        root, tmp_path / "catalog", object_ids, split_assignments=assignments,
        split_plan={"dataset_revision": revision})

    catalog = json.loads(catalog_path.read_text())
    assert catalog["evaluation_only"] is False and catalog["role"] == "train_validation"
    splits = json.loads((catalog_path.parent / "trainval-splits.json").read_text())
    assert sorted(splits["groups"].values()) == ["train", "validation"]
    assert splits["expected_object_counts"] == {"train": 1, "validation": 1}
    assert not (catalog_path.parent / "test-splits.json").exists()


def test_benchmark_catalog_is_the_only_20_object_identity_source(tmp_path, toy_spec):
    assets = []
    for index in range(20):
        asset_id = f"UPB_{index:08d}"
        spec_path = tmp_path / f"{asset_id}.json"
        spec_path.write_text(json.dumps(to_dict(dataclasses.replace(
            toy_spec, asset_id=asset_id, category=f"category-{index}"))))
        assets.append({"spec": spec_path.name, "split_group": f"uniphys:test:{asset_id}",
                       "origin": "fixture", "license": "fixture"})
    catalog = {"version": 1, "source_id": "uniphys-test", "dataset_revision": "a" * 40,
               "evaluation_only": True, "assets": assets}
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))

    _, _, specs = _load_frozen_catalog(catalog_path)
    assert [spec.asset_id for spec in specs] == sorted(spec.asset_id for spec in specs)
    assert specs[0].category == "category-0"

    catalog["assets"].pop()
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="Expected 20 catalog assets"):
        _load_frozen_catalog(catalog_path)
