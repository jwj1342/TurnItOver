import hashlib
import json

from PIL import Image

from turnitover.oracle.supervision import load_examples
from turnitover.teacher.export import export_episodes


def test_audited_teacher_episode_exports_and_loads(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    call = source / "model_calls/000"
    call.mkdir(parents=True)
    Image.new("RGB", (2, 2), "blue").save(source / "reference-00.png")
    (call / "images.json").write_text(json.dumps(["reference-00.png"]))
    (call / "prompt.txt").write_text("visible input")
    identity = {"formal_training": True, "split": "train", "reference_sha": "a" * 16,
                "candidate_sha": "b" * 16, "source_identity_sha256": "c" * 64,
                "source_content_sha256": "d" * 64}
    decision = {"verdict": {"status": "uncertain", "confidence": 0., "findings": [],
                             "summary": "Insufficient.", "limitations": ["One image."]}}
    episode = {"episode_id": "episode", "data_identity": identity,
               "teacher_verdict_missing": False,
               "termination": "teacher_finished", "decisions": [
                   {"call": 0, "valid": True, "decision": decision}],
               "model_calls": [{"path": "model_calls/000"}]}
    (source / "episode.json").write_text(json.dumps(episode))
    (source / "audit.json").write_text(json.dumps({
        "pipeline_ok": True, "teacher_verdict_present": True,
        "episode_sha256": hashlib.sha256((source / "episode.json").read_bytes()).hexdigest(),
        "independent_evidence_audit": {"accepted": True, "status": "uncertain",
                                        "rejection_reasons": []}}))

    manifest = export_episodes([source], output)
    loaded = list(load_examples(output, "train", "episodes",
                                require_formal_training=True, terminal_only=True))
    assert manifest["counts"]["diagnosis_accepted"] == 1
    assert len(loaded) == 1 and loaded[0][1][0].size == (2, 2)
    row = loaded[0][0]
    assert row["data_identity"] == identity
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
