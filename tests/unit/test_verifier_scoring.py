import json

import pytest

from turnitover.verifier.contracts import VERIFIER_SCHEMA_VERSION
from turnitover.verifier.gt_validator import GT_VALIDATION_SCHEMA_VERSION
from turnitover.verifier.scoring import score_verifier


def _finding(defect_id, parts, joint_id=None):
    return {"defect_id": defect_id, "parts": parts, "joint_id": joint_id}


def _gold(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _result(root, case_id, program_sha, status, findings, *, schema=VERIFIER_SCHEMA_VERSION):
    folder = root / case_id
    folder.mkdir(parents=True)
    value = {"schema_version": schema, "status": "complete", "program_sha": program_sha,
             "spent": 2, "trajectory": [{"step": 0}, {"step": 1}],
             "verdict": {"status": status, "findings": findings},
             "model_calls": [{"usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
                               "elapsed_ms": 5.0}]}
    (folder / "result.json").write_text(json.dumps(value))
    validation = {"schema_version": GT_VALIDATION_SCHEMA_VERSION, "status": "complete",
                  "candidate_sha": program_sha, "reference_sha": "reference",
                  "code_correct": not findings if status != "pass" else not bool(findings),
                  "checks": [], "privileged": True, "used_by_verifier": False}
    (folder / "gt_validation.json").write_text(json.dumps(validation))


def test_scores_detection_exact_joint_cost_and_latency(tmp_path):
    gold = tmp_path / "gold.jsonl"
    results = tmp_path / "results"
    rows = [
        {"id": "clean", "program_sha": "clean-sha",
         "target": {"status": "pass", "findings": []}},
        {"id": "axis-correct", "program_sha": "axis-correct-sha",
         "target": {"status": "fail", "findings": [
             _finding("kinematics.joint_axis", ["door"], "hinge")]}},
        {"id": "axis-wrong-joint", "program_sha": "axis-wrong-sha",
         "target": {"status": "fail", "findings": [
             _finding("kinematics.joint_axis", ["door"], "hinge")]}},
        {"id": "accepted-defect", "program_sha": "offset-sha",
         "target": {"status": "fail", "findings": [
             _finding("structure.part_offset", ["drawer"])]}},
    ]
    _gold(gold, rows)
    _result(results, "clean", "clean-sha", "pass", [])
    _result(results, "axis-correct", "axis-correct-sha", "fail",
            [_finding("kinematics.joint_axis", ["door"], "hinge")])
    _result(results, "axis-wrong-joint", "axis-wrong-sha", "fail",
            [_finding("kinematics.joint_axis", ["door"], "other-hinge")])
    _result(results, "accepted-defect", "offset-sha", "pass", [])
    validation = json.loads((results / "accepted-defect/gt_validation.json").read_text())
    validation["code_correct"] = False
    (results / "accepted-defect/gt_validation.json").write_text(json.dumps(validation))

    report = score_verifier(gold, results, tmp_path / "metrics.json")

    assert report["detection_accuracy"] == .75
    assert report["defect_recall"] == pytest.approx(2 / 3)
    assert report["false_accept_rate"] == pytest.approx(1 / 3)
    assert report["exact_match_accuracy"] == .5
    assert report["joint_localization_accuracy"] == .5
    assert report["average_observations"] == report["average_budget_spent"] == 2
    assert report["tokens"]["total"] == 40
    assert report["inference_time_ms"]["total"] == 20
    assert json.loads((tmp_path / "metrics.json").read_text())["status"] == "complete"


def test_missing_result_is_counted_as_error_not_dropped(tmp_path):
    gold = tmp_path / "gold.jsonl"
    _gold(gold, [{"id": "missing", "program_sha": "sha",
                  "target": {"status": "fail", "findings": [
                      _finding("structure.part_offset", ["door"])]}}])

    report = score_verifier(gold, tmp_path / "results", tmp_path / "metrics.json")

    assert report["n"] == 1 and report["error_rate"] == 1
    assert report["defect_recall"] == report["false_accept_rate"] == 0
    assert report["per_case"][0]["error"] == "missing_result"


def test_rejects_old_schema_and_candidate_mismatch(tmp_path):
    gold = tmp_path / "gold.jsonl"
    _gold(gold, [{"id": "case", "program_sha": "expected",
                  "target": {"status": "pass", "findings": []}}])
    _result(tmp_path / "old", "case", "expected", "pass", [], schema=1)
    with pytest.raises(ValueError, match="schema mismatch"):
        score_verifier(gold, tmp_path / "old", tmp_path / "old-metrics.json")
    _result(tmp_path / "wrong", "case", "different", "pass", [])
    with pytest.raises(ValueError, match="SHA mismatch"):
        score_verifier(gold, tmp_path / "wrong", tmp_path / "wrong-metrics.json")


def test_requires_validated_code_and_rejects_label_disagreement(tmp_path):
    gold = tmp_path / "gold.jsonl"
    row = {"id": "case", "program_sha": "sha",
           "target": {"status": "fail", "findings": [
               _finding("kinematics.joint_axis", ["door"], "hinge")]}}
    _gold(gold, [row])
    _result(tmp_path / "missing", "case", "sha", "fail", row["target"]["findings"])
    (tmp_path / "missing/case/gt_validation.json").unlink()

    report = score_verifier(gold, tmp_path / "missing", tmp_path / "missing-metrics.json")
    assert report["error_rate"] == report["gt_validation_error_rate"] == 1
    assert report["per_case"][0]["error"] == "missing_gt_validation"
    assert report["tokens"]["total"] == 10 and report["average_observations"] == 2

    _result(tmp_path / "wrong", "case", "sha", "fail", row["target"]["findings"])
    validation_path = tmp_path / "wrong/case/gt_validation.json"
    validation = json.loads(validation_path.read_text())
    validation["code_correct"] = True
    validation_path.write_text(json.dumps(validation))
    with pytest.raises(ValueError, match="Invalid ground truth"):
        score_verifier(gold, tmp_path / "wrong", tmp_path / "wrong-metrics.json")
