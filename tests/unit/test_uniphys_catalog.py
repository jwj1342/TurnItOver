import dataclasses
import hashlib
import json

from scripts import convert_uniphys
from turnitover.assets.catalog import CatalogSource


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
