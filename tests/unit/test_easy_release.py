import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from scripts.easy_release_archive import extract_zip, sha256, write_zip
from scripts import validate_easy_release as release
from scripts.validate_easy_release import verify_release_assets


def test_zip_is_reproducible_and_round_trips(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "readme.txt").write_text("deterministic fixture\n")
    (source / "nested").mkdir()
    (source / "nested/data.json").write_text('{"value": 1}\n')
    first = write_zip(source, tmp_path / "first.zip", "easy")
    os.utime(source / "readme.txt", (1000000000, 1000000000))
    second = write_zip(source, tmp_path / "second.zip", "easy")
    assert first == second
    extracted = extract_zip(tmp_path / "first.zip", tmp_path / "extract", first)
    assert (extracted / "nested/data.json").read_bytes() == (source / "nested/data.json").read_bytes()
    with pytest.raises(ValueError, match="must not exist"):
        extract_zip(tmp_path / "first.zip", tmp_path / "extract", first)


def test_pack_rejects_existing_output_and_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = tmp_path / "existing.zip"
    path.write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="must be new"):
        write_zip(source, path, "easy")
    assert path.read_bytes() == b"do not overwrite"
    (source / "link").symlink_to(path)
    with pytest.raises(ValueError, match="symbolic links"):
        write_zip(source, tmp_path / "new.zip", "easy")
    assert not (tmp_path / "new.zip").exists()


@pytest.mark.parametrize("name,mode", [
    ("easy/../../escape", stat.S_IFREG),
    ("/easy/escape", stat.S_IFREG),
    ("other/escape", stat.S_IFREG),
    ("easy\\escape", stat.S_IFREG),
    ("easy/link", stat.S_IFLNK),
    ("easy/fifo", stat.S_IFIFO),
])
def test_extraction_rejects_unsafe_members_before_writing(tmp_path, name, mode):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo(name)
        info.create_system = 3
        info.external_attr = (mode | 0o644) << 16
        archive.writestr(info, b"unsafe")
    expected = dict(root="easy", sha256=sha256(path), size=path.stat().st_size,
                    files=1, uncompressed_bytes=6)
    output = tmp_path / "extract"
    with pytest.raises(ValueError, match="unsafe ZIP"):
        extract_zip(path, output, expected)
    assert not output.exists()


@pytest.fixture
def assets(tmp_path):
    destination = tmp_path / "assets"
    destination.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.txt").write_text("synthetic fixture")
    review = source / "validation_review"
    review.mkdir()
    (review / "example.txt").write_text("synthetic blind fixture")
    contract = dict(version=1, tag="fixture", expected_report={"cases": 1},
                    preserved_files={"example.txt": sha256(source / "example.txt")}, assets={})
    for name, root in (("data.zip", "easy"), ("review.zip", "validation_review")):
        contract["assets"][name] = write_zip(source if root == "easy" else review,
                                             destination / name, root)
    manifest = tmp_path / "pinned.json"
    manifest.write_text(json.dumps(contract) + "\n")
    (destination / "release-manifest.json").write_bytes(manifest.read_bytes())
    def refresh_checksums():
        names = sorted([*contract["assets"], "release-manifest.json"])
        (destination / "SHA256SUMS").write_text("".join(f"{sha256(destination / name)}  {name}\n" for name in names))
    refresh_checksums()
    return destination, manifest, contract, refresh_checksums


def test_pinned_assets_verify_without_dataset_or_network(assets):
    directory, manifest, contract, _ = assets
    assert verify_release_assets(directory, manifest) == contract


def test_missing_release_asset_fails(assets):
    directory, manifest, _, _ = assets
    (directory / "data.zip").unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_release_assets(directory, manifest)


def test_checksum_file_alone_cannot_authorize_changed_assets(assets):
    directory, manifest, _, refresh = assets
    with (directory / "data.zip").open("ab") as stream:
        stream.write(b"tampered")
    refresh()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_release_assets(directory, manifest)


def test_downloaded_manifest_must_match_code_contract(assets):
    directory, manifest, _, refresh = assets
    (directory / "release-manifest.json").write_text("{}\n")
    refresh()
    with pytest.raises(ValueError, match="code-pinned"):
        verify_release_assets(directory, manifest)


def test_no_unlisted_release_assets(assets):
    directory, manifest, _, _ = assets
    (directory / "filesystem.tar.gz").write_bytes(b"must not publish a construction archive")
    with pytest.raises(ValueError, match="unexpected"):
        verify_release_assets(directory, manifest)


@pytest.fixture
def validated_assets(assets, monkeypatch):
    directory, manifest, contract, _ = assets

    def validate_dataset(root):
        assert (root / "example.txt").read_text() == "synthetic fixture"
        return contract["expected_report"]

    def validate_review(root):
        assert (root / "example.txt").read_text() == "synthetic blind fixture"

    monkeypatch.setattr(release, "validate_bundle", validate_dataset)
    monkeypatch.setattr(release, "validate_review_bundle", validate_review)
    return directory, manifest, contract


def test_validated_release_publishes_both_roots(validated_assets, tmp_path):
    directory, manifest, contract = validated_assets
    output = tmp_path / "accepted"
    report = release.validate_release(directory, manifest, output)
    assert report["dataset"] == contract["expected_report"]
    assert (output / "easy/example.txt").read_text() == "synthetic fixture"
    assert (output / "validation_review/example.txt").read_text() == "synthetic blind fixture"
    assert not list(tmp_path.glob(".easy-release-*"))


@pytest.mark.parametrize("kind", ["empty_directory", "nonempty_directory", "file", "dangling_symlink"])
def test_release_preserves_preexisting_output(validated_assets, tmp_path, kind):
    directory, manifest, _ = validated_assets
    output = tmp_path / "existing"
    if kind.endswith("directory"):
        output.mkdir()
        if kind == "nonempty_directory":
            (output / "sentinel").write_text("keep")
    elif kind == "file":
        output.write_text("keep")
    else:
        output.symlink_to(tmp_path / "missing", target_is_directory=True)
    identity = output.lstat()
    with pytest.raises(ValueError, match="must not exist"):
        release.validate_release(directory, manifest, output)
    assert output.lstat().st_ino == identity.st_ino
    if kind == "nonempty_directory":
        assert (output / "sentinel").read_text() == "keep"
    elif kind == "file":
        assert output.read_text() == "keep"


@pytest.mark.parametrize("nonempty", [False, True])
def test_release_preserves_directory_created_during_validation(validated_assets, tmp_path, monkeypatch, nonempty):
    directory, manifest, contract = validated_assets
    output = tmp_path / "competing"
    identities = []

    def validate_dataset(root):
        output.mkdir()
        identities.append(output.stat().st_ino)
        if nonempty:
            (output / "sentinel").write_text("keep")
        return contract["expected_report"]

    monkeypatch.setattr(release, "validate_bundle", validate_dataset)
    with pytest.raises(FileExistsError):
        release.validate_release(directory, manifest, output)
    assert output.stat().st_ino == identities[0]
    assert sorted(path.name for path in output.iterdir()) == (["sentinel"] if nonempty else [])
    assert not list(tmp_path.glob(".easy-release-*"))


def test_publication_failure_removes_partial_output(validated_assets, tmp_path, monkeypatch):
    directory, manifest, _ = validated_assets
    output = tmp_path / "failed-publication"
    rename = Path.rename
    moved = []

    def fail_second_move(source, target):
        if source.parent.name == "validated":
            if moved:
                raise OSError("injected publication failure")
            moved.append(source.name)
        return rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_second_move)
    with pytest.raises(OSError, match="injected publication failure"):
        release.validate_release(directory, manifest, output)
    assert moved == ["easy"]
    assert not output.exists()
    assert not list(tmp_path.glob(".easy-release-*"))


@pytest.mark.parametrize("step", ["extract_zip", "validate_bundle", "validate_review_bundle"])
def test_validation_failures_never_publish_output(validated_assets, tmp_path, monkeypatch, step):
    directory, manifest, _ = validated_assets
    output = tmp_path / "failed-validation"

    def fail(*args):
        raise ValueError(f"injected {step} failure")

    monkeypatch.setattr(release, step, fail)
    with pytest.raises(ValueError, match=f"injected {step} failure"):
        release.validate_release(directory, manifest, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".easy-release-*"))
