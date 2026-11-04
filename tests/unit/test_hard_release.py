import json
import pytest

from scripts.build_hard_release import build_release
from scripts.easy_release_archive import sha256
from scripts.validate_hard_bundle import validate_bundle
from scripts.validate_hard_release import validate_release
from tests.unit.hard_bundle_fixture import build_bundle


def test_hard_release_is_reproducible_and_portable(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    assert validate_bundle(root)["native_images"] == 900
    first = tmp_path / "first"
    second = tmp_path / "second"
    contract = build_release(root, first, "hard-candidate-v0.1.0-rc.1", "1" * 40)
    build_release(root, second, contract["tag"], "1" * 40)
    assert {p.name: sha256(p) for p in first.iterdir()} == {
        p.name: sha256(p) for p in second.iterdir()
    }
    report = validate_release(
        first, first / "release-manifest.json", tmp_path / "accepted"
    )
    assert (
        report["dataset"]["objects"] == 75
        and report["dataset"]["training_eligible_cases"] == 0
    )
    assert not (tmp_path / "accepted/hard/private/programs").exists()


def test_hard_release_rejects_changed_zip(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    assets = tmp_path / "assets"
    contract = build_release(root, assets, "hard-candidate-v0.1.0-rc.1", "1" * 40)
    path = assets / next(iter(contract["assets"]))
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="size or SHA-256 mismatch"):
        validate_release(
            assets, assets / "release-manifest.json", tmp_path / "rejected"
        )


def test_hard_bundle_rejects_absolute_path_and_answer_leak(tmp_path):
    root = build_bundle(tmp_path / "bundle")
    gold = [
        json.loads(line)
        for line in (root / "private/gold.jsonl").read_text().splitlines()
    ]
    gold[0]["source"]["dataset"] = "/tmp/private-dataset"
    (root / "private/gold.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in gold)
    )
    from scripts.hard_bundle_inventory import write_manifest

    write_manifest(root)
    with pytest.raises(ValueError, match="absolute path"):
        validate_bundle(root)
