"""Snapshot reproducibility and corruption checks without model dependencies."""
import copy
import hashlib
import json
import shutil

import pytest

from turnitover.oracle.evaluation_snapshot import create_snapshot, load_snapshot


def inputs(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text('{"version": "frozen"}')
    (dataset / "annotations.private.jsonl").write_text(
        '{"id": "case-a", "labels": [{"defect_id": "private-defect"}]}\n')
    image = b"same image bytes"
    for name in ("a.png", "b.png"):
        (dataset / name).write_bytes(image)
    record = {"id": "case-a:terminal", "episode_id": "case-a",
              "images": ["a.png", "b.png"],
              "image_sha256": [hashlib.sha256(image).hexdigest()] * 2,
              "messages": [{"role": "user", "content": "Public prompt only"}],
              "evidence": {"covered_label_indices": [0]}}
    return dataset, [record]


def freeze(tmp_path):
    dataset, records = inputs(tmp_path)
    output = tmp_path / "evaluation"
    return dataset, records, output, create_snapshot(
        output, dataset, records, "validation", "diagnosis", ["private-defect"])


@pytest.mark.parametrize("change", ["delete", "update"])
def test_snapshot_survives_original_dataset_change(tmp_path, change):
    dataset, records, output, snapshot = freeze(tmp_path)
    if change == "delete":
        shutil.rmtree(dataset)
    else:
        (dataset / "a.png").write_bytes(b"changed")
        (dataset / "annotations.private.jsonl").write_text("broken")
    restored = load_snapshot(output)
    assert restored["records"] == snapshot["records"]
    assert restored["annotations_by_id"]["case-a"]["labels"]
    assert restored["records"][0]["messages"] == records[0]["messages"]
    assert "private-defect" not in json.dumps(restored["records"][0]["messages"])
    assert records[0]["images"] == ["a.png", "b.png"]


def test_images_deduplicate_by_content_and_original_paths_are_preserved(tmp_path):
    _, _, _, snapshot = freeze(tmp_path)
    assert len(list((snapshot["root"] / "images").iterdir())) == 1
    assert len(snapshot["records"][0]["images"]) == 2
    assert snapshot["image_paths"][0]["original_images"] == ["a.png", "b.png"]
    assert snapshot["manifest"]["case_count"] == 1
    assert snapshot["manifest"]["task"] == "diagnosis"
    assert snapshot["manifest"]["split"] == "validation"


@pytest.mark.parametrize("name", ["records.jsonl", "annotations.private.jsonl", "dataset-manifest.json", "image-paths.jsonl", "image"])
def test_modified_snapshot_file_is_rejected(tmp_path, name):
    _, _, output, snapshot = freeze(tmp_path)
    relative = snapshot["records"][0]["images"][0] if name == "image" else name
    (snapshot["root"] / relative).write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(output)


def test_missing_image_is_rejected(tmp_path):
    _, _, output, snapshot = freeze(tmp_path)
    (snapshot["root"] / snapshot["records"][0]["images"][0]).unlink()
    with pytest.raises(FileNotFoundError):
        load_snapshot(output)


def test_duplicate_record_ids_rejected_before_snapshot_creation(tmp_path):
    dataset, records = inputs(tmp_path)
    with pytest.raises(ValueError, match="duplicate record"):
        create_snapshot(tmp_path / "evaluation", dataset, records * 2,
                        "validation", "diagnosis", [])
    assert not (tmp_path / "evaluation" / "scoring-snapshot").exists()


@pytest.mark.parametrize("kind", ["missing", "hash", "escape"])
def test_invalid_source_image_has_no_completed_manifest(tmp_path, kind):
    dataset, records = inputs(tmp_path)
    if kind == "missing":
        (dataset / "a.png").unlink()
    elif kind == "hash":
        records[0]["image_sha256"][0] = "0" * 64
    else:
        records[0]["images"][0] = "../outside.png"
    with pytest.raises((ValueError, FileNotFoundError)):
        create_snapshot(tmp_path / "evaluation", dataset, records, "validation", "diagnosis", [])
    assert not (tmp_path / "evaluation" / "scoring-snapshot" / "manifest.json").exists()


def test_loader_rejects_duplicate_ids_even_with_updated_file_hash(tmp_path):
    _, _, output, snapshot = freeze(tmp_path)
    path = snapshot["root"] / "records.jsonl"
    path.write_bytes(path.read_bytes() * 2)
    manifest = copy.deepcopy(snapshot["manifest"])
    manifest["files"]["records.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["case_count"] = 2
    (snapshot["root"] / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate record"):
        load_snapshot(output)


def test_annotations_only_include_evaluated_episodes_and_keep_source_hashes(tmp_path):
    dataset, records = inputs(tmp_path)
    annotation_path = dataset / "annotations.private.jsonl"
    with annotation_path.open("a") as stream:
        stream.write('{"id": "sealed-test", "split": "test", "labels": ["sealed-label"]}\n')
        stream.write('{"id": "other-validation", "split": "validation", "labels": []}\n')
    original_bytes = annotation_path.read_bytes()
    snapshot = create_snapshot(tmp_path / "evaluation", dataset, records, "validation", "diagnosis", [])
    assert set(snapshot["annotations_by_id"]) == {"case-a"}
    assert b"sealed-label" not in (snapshot["root"] / "annotations.private.jsonl").read_bytes()
    assert snapshot["manifest"]["source_sha256"] == {
        "dataset_manifest": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "annotations": hashlib.sha256(original_bytes).hexdigest(),
    }


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong-split"])
def test_invalid_annotations_rejected_before_completed_snapshot(tmp_path, kind):
    dataset, records = inputs(tmp_path)
    path = dataset / "annotations.private.jsonl"
    if kind == "missing":
        path.write_text('{"id": "other-case", "labels": []}\n')
    elif kind == "duplicate":
        path.write_bytes(path.read_bytes() * 2)
    else:
        path.write_text('{"id": "case-a", "split": "test", "labels": []}\n')
    with pytest.raises(ValueError):
        create_snapshot(tmp_path / "evaluation", dataset, records, "validation", "diagnosis", [])
    assert not (tmp_path / "evaluation" / "scoring-snapshot" / "manifest.json").exists()
