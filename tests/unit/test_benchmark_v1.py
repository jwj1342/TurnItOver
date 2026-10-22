import json
from pathlib import Path

import pytest

from turnitover.benchmark.manifest import (build_artifacts, condition_inputs, load_gold, load_samples,
                                           sha256_file, verify_freeze)
from turnitover.benchmark.protocol import CONDITIONS, InvalidOutput, parse_prediction
from turnitover.benchmark.scoring import score

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark" / "v1"
JOINTS, PARTS = {"joint_0"}, {"link_0"}


def _write(tmp_path, rows, name="p.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _predictions(split, make):
    return [make(sid, g) for sid, g in sorted(load_gold(BENCH, split).items())]


def _oracle(sid, g, evidence=(1, 3)):
    return {"case_id": sid, "decision": g["status"],
            "findings": [{"defect_id": d["defect_id"], "joint_id": d["joint_id"], "parts": d["parts"],
                          "evidence": list(evidence)} for d in g["defects"]]}


def test_loader_reads_nonzero_real_samples_and_verifies_hashes():
    rows = load_samples(ROOT, BENCH)
    assert len(rows) == 120 and {r["split"] for r in rows} == {"train", "validation"}
    assert all(len(r["images"]) == 4 for r in rows)
    verify_freeze(BENCH)


def test_frozen_files_are_reproducible_from_evidence():
    frozen = build_artifacts(ROOT)
    for name, text in frozen.items():
        assert (BENCH / name).read_text(encoding="utf-8") == text, name


def test_train_validation_have_no_object_or_source_overlap():
    split = json.loads((BENCH / "split.json").read_text(encoding="utf-8"))["objects"]
    by = {s: [o for o in split if o["split"] == s] for s in ("train", "validation")}
    assert by["validation"] and by["train"]
    for key in ("asset_id", "split_group", "source_model_id", "source_archive_sha256", "geometry_fingerprint"):
        assert {o[key] for o in by["train"]}.isdisjoint({o[key] for o in by["validation"]}), key
    assert not any(o["exposed_to_prior_experiments"] for o in by["validation"])


def test_conditions_never_expose_gold():
    row = load_samples(ROOT, BENCH, "validation")[0]
    for name in CONDITIONS:
        visible = condition_inputs(row, name)
        assert not {"status", "defects", "review"} & set(visible)
    assert "images" not in condition_inputs(row, "no_image") and "code" not in condition_inputs(row, "rgb")


def test_parse_prediction_is_strict():
    ok = {"case_id": "a", "decision": "fail", "findings": [
        {"defect_id": "kinematics.joint_axis", "joint_id": "joint_0", "parts": ["link_0"], "evidence": [1, 3]}]}
    assert parse_prediction(ok, condition="rgb", joint_ids=JOINTS, part_ids=PARTS).decision == "fail"
    bad = [
        {**ok, "findings": []},
        {**ok, "decision": "pass"},
        {**ok, "extra": 1},
        {**ok, "findings": [{**ok["findings"][0], "defect_id": "made.up"}]},
        {**ok, "findings": [{**ok["findings"][0], "joint_id": "joint_9"}]},
        {**ok, "findings": [{**ok["findings"][0], "evidence": [4]}]},
        {**ok, "findings": [{**ok["findings"][0], "evidence": []}]},
    ]
    for item in bad:
        with pytest.raises(InvalidOutput):
            parse_prediction(item, condition="rgb", joint_ids=JOINTS, part_ids=PARTS)
    with pytest.raises(InvalidOutput):
        parse_prediction(ok, condition="no_image", joint_ids=JOINTS, part_ids=PARTS)


def test_oracle_scores_perfectly_and_is_byte_reproducible(tmp_path):
    path = _write(tmp_path, _predictions("validation", _oracle))
    first = score(BENCH, path, tmp_path / "a.json", condition="rgb")
    score(BENCH, path, tmp_path / "b.json", condition="rgb")
    assert sha256_file(tmp_path / "a.json") == sha256_file(tmp_path / "b.json")
    m = first["metrics"]
    assert m["macro_f1"] == m["localization_f1"] == m["exact_verdict"] == m["evidence_consistency"] == 1.0
    assert m["false_pass"] == 0.0 and m["invalid_output"] == 0.0 and first["n"] == 24


def test_always_pass_is_all_false_pass(tmp_path):
    rows = _predictions("validation", lambda sid, g: {"case_id": sid, "decision": "pass", "findings": []})
    m = score(BENCH, _write(tmp_path, rows), tmp_path / "s.json", condition="rgb")["metrics"]
    assert m["false_pass"] == 1.0 and m["exact_verdict"] == 0.5 and m["evidence_consistency"] is None


def test_missing_and_invalid_rows_count_as_invalid(tmp_path):
    rows = _predictions("validation", _oracle)
    rows = rows[:-2] + [{**rows[-2], "decision": "maybe"}]
    report = score(BENCH, _write(tmp_path, rows), tmp_path / "s.json", condition="rgb")
    assert report["metrics"]["invalid_output"] == 2 / 24


def test_wrong_evidence_breaks_consistency_not_localization(tmp_path):
    rows = _predictions("validation", lambda sid, g: _oracle(sid, g, evidence=(0,)))
    m = score(BENCH, _write(tmp_path, rows), tmp_path / "s.json", condition="rgb")["metrics"]
    assert m["evidence_consistency"] == 0.0 and m["localization_f1"] == 1.0


def test_insufficient_evidence_condition_expects_abstention(tmp_path):
    abstain = _predictions("validation", lambda sid, g: {"case_id": sid, "decision": "continue", "findings": []})
    report = score(BENCH, _write(tmp_path, abstain), tmp_path / "a.json", condition="no_image")
    assert report["metrics"]["exact_verdict"] == 1.0 and report["supporting"]["abstain_correct_rate"] == 1.0
    guess = _predictions("validation", lambda sid, g: {"case_id": sid, "decision": "pass", "findings": []})
    assert score(BENCH, _write(tmp_path, guess, "g.jsonl"), tmp_path / "g.json",
                 condition="no_image")["metrics"]["exact_verdict"] == 0.0


def test_scoring_rejects_foreign_samples_and_existing_output(tmp_path):
    train_id = next(iter(load_gold(BENCH, "train")))
    with pytest.raises(ValueError):
        score(BENCH, _write(tmp_path, [{"case_id": train_id, "decision": "pass", "findings": []}]),
              tmp_path / "s.json", condition="rgb")
    (tmp_path / "exists.json").write_text("{}")
    with pytest.raises(ValueError):
        score(BENCH, _write(tmp_path, [], "e.jsonl"), tmp_path / "exists.json", condition="rgb")
