import hashlib
import json
from pathlib import Path

from PIL import Image

from turnitover.oracle.supervision import load_examples
from turnitover.teacher.export import _copy_images, export_episodes


def _source(root: Path, color: str, *, episode_id="episode", experiment_mode="active") -> Path:
    source = root
    call = source / "model_calls/000"
    call.mkdir(parents=True)
    Image.new("RGB", (2, 2), color).save(source / "reference-00.png")
    (call / "images.json").write_text(json.dumps(["reference-00.png"]))
    (call / "prompt.txt").write_text("visible input")
    identity = {"formal_training": True, "split": "train", "reference_sha": "a" * 16,
                "candidate_sha": "b" * 16, "source_identity_sha256": "c" * 64,
                "source_content_sha256": "d" * 64, "experiment_mode": experiment_mode}
    decision = {"verdict": {"status": "uncertain", "confidence": 0., "findings": [],
                             "summary": "Insufficient.", "limitations": ["One image."]}}
    episode = {"episode_id": episode_id, "data_identity": identity,
               "teacher_verdict_missing": False, "termination": "teacher_finished",
               "decisions": [{"call": 0, "valid": True, "decision": decision}],
               "model_calls": [{"path": "model_calls/000"}]}
    (source / "episode.json").write_text(json.dumps(episode))
    (source / "audit.json").write_text(json.dumps({
        "pipeline_ok": True, "teacher_verdict_present": True,
        "episode_sha256": hashlib.sha256((source / "episode.json").read_bytes()).hexdigest(),
        "independent_evidence_audit": {"accepted": True, "status": "uncertain",
                                        "diagnosis_objective": "first_failure",
                                        "rejection_reasons": []}}))
    return source


def test_audited_teacher_episode_exports_and_loads(tmp_path):
    source, output = _source(tmp_path / "source", "blue"), tmp_path / "export"
    identity = json.loads((source / "episode.json").read_text())["data_identity"]

    manifest = export_episodes([source], output)
    loaded = list(load_examples(output, "train", "episodes",
                                require_formal_training=True, terminal_only=True))
    assert manifest["counts"]["diagnosis_accepted"] == 1
    assert len(loaded) == 1 and loaded[0][1][0].size == (2, 2)
    row = loaded[0][0]
    assert row["data_identity"] == identity
    assert row["source_episode"] == str(source)
    assert row["image_provenance"][0]["source"] == "reference-00.png"
    assert row["image_sha256"] == [hashlib.sha256(
        (output / row["images"][0]).read_bytes()).hexdigest()]


def test_export_rejects_semantically_accepted_episode_when_integrity_failed(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    source.mkdir()
    (source / "episode.json").write_text(json.dumps({
        "episode_id": "bad", "teacher_verdict_missing": False, "decisions": [], "model_calls": []}))
    (source / "audit.json").write_text(json.dumps({
        "pipeline_ok": False, "teacher_verdict_present": True,
        "independent_evidence_audit": {"accepted": True, "rejection_reasons": []}}))
    manifest = export_episodes([source], output)
    assert manifest["counts"]["episodes_blocked"] == 1
    rejected = json.loads((output / "rejected.jsonl").read_text())
    assert "pipeline_integrity_audit_failed" in rejected["reasons"]


def test_red_blue_same_episode_and_call_never_reuse_the_wrong_image(tmp_path):
    red = _source(tmp_path / "condition-red", "red")
    blue = _source(tmp_path / "condition-blue", "blue")
    output = tmp_path / "export"
    export_episodes([red, blue], output)
    rows = [json.loads(line) for line in (output / "train.diagnosis.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and len({row["id"] for row in rows}) == 2
    assert len({row["images"][0] for row in rows}) == 2
    assert [Image.open(output / row["images"][0]).getpixel((0, 0)) for row in rows] == [
        (255, 0, 0), (0, 0, 255)]
    loaded = list(load_examples(output, "train", "diagnosis", require_formal_training=True))
    assert [images[0].getpixel((0, 0)) for _, images in loaded] == [(255, 0, 0), (0, 0, 255)]
    assert all(row["image_sha256"][0] == row["image_provenance"][0]["source_sha256"]
               for row in rows)


def test_multiple_episodes_conditions_and_repeat_exports_preserve_image_identity(tmp_path):
    sources = [_source(tmp_path / "run-a", "red", episode_id="one", experiment_mode="active"),
               _source(tmp_path / "run-b", "blue", episode_id="one", experiment_mode="fixed"),
               _source(tmp_path / "run-c", "green", episode_id="two", experiment_mode="active")]
    hashes = []
    for name in ("export-a", "export-b"):
        output = tmp_path / name
        export_episodes(sources, output)
        rows = [json.loads(line) for line in (output / "train.diagnosis.jsonl").read_text().splitlines()]
        hashes.append([(row["id"], row["image_sha256"], row["source_run_id"]) for row in rows])
        assert all(hashlib.sha256((output / row["images"][0]).read_bytes()).hexdigest()
                   == row["image_sha256"][0] for row in rows)
    assert hashes[0] == hashes[1]


def test_existing_corrupt_destination_is_safely_rebuilt(tmp_path):
    source = _source(tmp_path / "source", "blue")
    output = tmp_path / "export"
    source_sha = hashlib.sha256((source / "reference-00.png").read_bytes()).hexdigest()
    destination = output / "images/episode" / f"000-00-{source_sha}.png"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"wrong pre-existing bytes")
    paths, hashes, provenance = _copy_images(
        source, output, "episode", {"path": "model_calls/000"})
    assert paths == ["images/episode/" + destination.name]
    assert hashes == [source_sha] and provenance[0]["exported_sha256"] == source_sha
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == source_sha
