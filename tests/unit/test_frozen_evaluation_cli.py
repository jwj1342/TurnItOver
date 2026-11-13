"""Offline CLI scoring stays bound to frozen inputs and complete predictions."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.unit.test_snapshot_scoring import AXIS, OFFSET, inputs
from turnitover.oracle.evaluation_snapshot import create_snapshot
from turnitover.oracle.sft_metrics import public_input_digest


CLI = Path(__file__).resolve().parents[2] / "scripts" / "score_frozen_evaluation.py"
ROTATION = "structure.part_rotation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(tmp_path, *, count=1, taxonomy=(OFFSET, AXIS), name="run"):
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    image = dataset / "image.png"
    image.write_bytes(b"frozen image content; no image decoding during scoring")
    records, annotations, predictions = [], [], []
    for index in range(count):
        source_records, source_annotations, source_predictions = inputs()
        record, annotation, prediction = source_records[0], source_annotations[0], source_predictions[0]
        uid = f"case-{index}"
        identity = {"source_dataset": "synthetic", "source_asset_id": uid,
                    "source_identity_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                    "source_content_sha256": "c" * 64, "formal_training": True,
                    "reference_sha": "d" * 64, "candidate_sha": "e" * 64}
        record.update(id=uid + ":verdict", episode_id=uid, data_identity=identity,
                      images=["image.png"] * 6, image_sha256=[sha(image)] * 6)
        annotation.update(id=uid, split="validation", data_identity=copy.deepcopy(identity),
                          reference_sha=identity["reference_sha"], candidate_sha=identity["candidate_sha"])
        annotation["labels"][1]["defect_id"] = taxonomy[1]
        if taxonomy[1] != AXIS:
            annotation["labels"][1]["params"] = {}
        prediction["id"] = record["id"]
        records.append(record)
        annotations.append(annotation)
        predictions.append(prediction)
    write_json(dataset / "manifest.json", {"dataset_version": "synthetic-v1",
                                           "experiment_taxonomy": list(taxonomy),
                                           "counts": {"validation": count}})
    # Sealed labels establish that the snapshot is filtered, while protocol
    # provenance deliberately identifies the complete source annotation file.
    source_annotations = annotations + [{"id": "sealed-test", "split": "test", "labels": []}]
    (dataset / "annotations.private.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in source_annotations), encoding="utf-8")
    evaluation = tmp_path / name
    snapshot = create_snapshot(evaluation, dataset, records, "validation", "diagnosis", taxonomy)
    prediction_path = evaluation / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    protocol = {"task": "diagnosis", "split": "validation", "n": count,
                "defect_ids": list(taxonomy), "dataset_version": "synthetic-v1",
                "dataset_manifest_sha256": sha(dataset / "manifest.json"),
                "annotations_sha256": sha(dataset / "annotations.private.jsonl"),
                "public_input_sha256": public_input_digest(records)}
    write_json(evaluation / "metrics.json", {
        "status": "complete", "protocol": protocol,
        "scoring_snapshot_sha256": sha(snapshot["root"] / "manifest.json"),
        "predictions_sha256": sha(prediction_path)})
    assert snapshot["manifest"]["files"]["annotations.private.jsonl"] != protocol["annotations_sha256"]
    return dataset, evaluation


def run_cli(evaluation, output):
    return subprocess.run([sys.executable, str(CLI), "--evaluation", str(evaluation),
                           "--out", str(output)], capture_output=True, text=True)


@pytest.mark.parametrize("change", ["delete", "update"])
def test_offline_cli_uses_snapshot_after_source_dataset_changes(tmp_path, change):
    dataset, evaluation = make_run(tmp_path)
    if change == "delete":
        shutil.rmtree(dataset)
    else:
        for path in dataset.iterdir():
            path.write_bytes(b"updated source no longer matches frozen evaluation")
    output = tmp_path / "offline-metrics.json"
    result = run_cli(evaluation, output)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    assert report["n"] == 1
    assert report["paper_metrics"]["strict_evidence_support_rate"] == 1
    assert report["paper_metrics"]["all_injected_metrics"]["overall"]["defect_per_class"][AXIS]["fn"] == 1
    assert report["predictions_sha256"] == sha(evaluation / "predictions.jsonl")


def test_new_snapshot_tracks_updated_case_count_and_taxonomy(tmp_path):
    _, old_evaluation = make_run(tmp_path, name="old-run")
    _, new_evaluation = make_run(tmp_path, count=2, taxonomy=(OFFSET, ROTATION), name="new-run")
    for evaluation, count, labels in ((old_evaluation, 1, [OFFSET, AXIS]),
                                       (new_evaluation, 2, [OFFSET, ROTATION])):
        output = tmp_path / (evaluation.name + ".json")
        result = run_cli(evaluation, output)
        assert result.returncode == 0, result.stderr
        report = json.loads(output.read_text())
        assert report["n"] == count
        assert report["defect_labels"] == sorted(labels)
        assert report["paper_metrics"]["all_injected_metrics"]["overall"]["defect_labels"] == sorted(labels)


def test_cli_rejects_changed_prediction_bytes(tmp_path):
    _, evaluation = make_run(tmp_path)
    with (evaluation / "predictions.jsonl").open("a") as stream:
        stream.write("\n")
    output = tmp_path / "rejected.json"
    result = run_cli(evaluation, output)
    assert result.returncode != 0
    assert "hash mismatch" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("mode", ["missing", "duplicate", "extra"])
def test_cli_rejects_inexact_prediction_coverage_even_with_updated_hash(tmp_path, mode):
    _, evaluation = make_run(tmp_path)
    prediction_path = evaluation / "predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    if mode == "missing":
        rows.clear()
    elif mode == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows.append({"id": "extra", "raw": "invalid"})
    prediction_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metrics = json.loads((evaluation / "metrics.json").read_text())
    metrics["predictions_sha256"] = sha(prediction_path)
    write_json(evaluation / "metrics.json", metrics)
    output = tmp_path / "rejected.json"
    result = run_cli(evaluation, output)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("mode", ["incomplete", "missing-prediction-file", "protocol-count", "protocol-taxonomy"])
def test_cli_rejects_incomplete_or_mismatched_runs(tmp_path, mode):
    _, evaluation = make_run(tmp_path)
    metrics = json.loads((evaluation / "metrics.json").read_text())
    if mode == "incomplete":
        metrics["status"] = "incomplete"
    elif mode == "missing-prediction-file":
        (evaluation / "predictions.jsonl").unlink()
    elif mode == "protocol-count":
        metrics["protocol"]["n"] = 2
    else:
        metrics["protocol"]["defect_ids"] = [OFFSET, ROTATION]
    write_json(evaluation / "metrics.json", metrics)
    output = tmp_path / "rejected.json"
    result = run_cli(evaluation, output)
    assert result.returncode != 0
    assert not output.exists()
