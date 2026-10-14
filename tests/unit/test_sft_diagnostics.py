import json

import pytest

from scripts.audit_sft_supervision import _integrity
from scripts.analyze_qwen_episode_diagnostics import _stop_metrics, summarize, summarize_defect
from scripts.evaluate_budget_baseline import MajorityLookup, features
from scripts.evaluate_qwen_episode_diagnostics import remove_remaining, visible_findings
from scripts.evaluate_qwen_stopping import _parse, _summarize


def test_visible_findings_uses_only_causal_witness_prefix():
    annotation = {
        "labels": [
            {"defect_id": "structure.part_offset", "parts": ["door"], "params": {}},
            {"defect_id": "kinematics.joint_axis", "parts": ["door"],
             "params": {"joint": "hinge"}},
        ],
        "exposures": [[0], [1]],
        "trajectories": {"oracle": {
            "history": [{"step": 0}, {"step": 1}], "witness_indices": [0, 1],
        }},
    }
    assert visible_findings(annotation, 1) == {
        ("structure.part_offset", None, ("door",)): {0},
    }


def test_remove_remaining_preserves_real_history_and_other_payload_fields():
    payload = {"context": {"task": "x"}, "remaining": 4,
               "history": [{"step": 0}], "image_order": ["a", "b"]}
    record = {"messages": [{"role": "user", "content": "protocol\n" + json.dumps(payload)},
                            {"role": "assistant", "content": "{}"}]}
    changed = remove_remaining(record)
    changed_payload = json.loads(changed["messages"][0]["content"].rsplit("\n", 1)[-1])
    assert "remaining" not in changed_payload
    assert changed_payload["history"] == payload["history"]
    assert record["messages"][0]["content"].endswith(json.dumps(payload))


def test_diagnostic_summary_counts_exact_and_unsupported_findings():
    expected = [["structure.part_offset", None, ["door"]]]
    rows = [
        {"decision_kind": "verdict", "predicted_status": "fail", "expected_status": "fail",
         "expected_findings": expected, "predicted_findings": expected, "unsupported_findings": 0},
        {"decision_kind": "verdict", "predicted_status": "fail", "expected_status": "uncertain",
         "expected_findings": [], "predicted_findings": expected, "unsupported_findings": 1},
    ]
    result = summarize(rows)
    assert result["exact_verdict"] == 1
    assert result["findings"] == {"tp": 1, "fp": 1, "fn": 0,
                                  "precision": .5, "recall": 1., "f1": 2 / 3}
    assert result["false_fail"] == 1
    assert result["unsupported_findings"] == 1
    assert summarize_defect(rows, "structure.part_offset") == {
        "tp": 1, "fp": 1, "fn": 0, "precision": .5, "recall": 1., "f1": 2 / 3,
        "expected_cases": 1, "predicted_cases": 2,
    }


def test_budget_baseline_uses_only_declared_metadata_features():
    payload = {"remaining": 4, "history": [{"step": 0}, {"step": 1}]}
    record = {"decision_index": 2, "images": ["ref", "a", "b"],
              "messages": [{"content": "protocol\n" + json.dumps(payload)}]}

    assert features(record) == (2, 4, 2, 3)
    model = MajorityLookup()
    model.fit_one(features(record), "fail")
    assert model.predict((2, 4, 2, 3)) == "fail"
    assert model.predict((9, 0, 9, 10)) == "fail"


def test_stop_metrics_separate_prefix_accuracy_from_first_stop_timing():
    rows = [
        {"episode_id": "a", "decision_index": 0, "decision_kind": "action"},
        {"episode_id": "a", "decision_index": 1, "decision_kind": "verdict"},
        {"episode_id": "a", "decision_index": 2, "decision_kind": "verdict"},
    ]
    targets = {("a", 0): "continue", ("a", 1): "continue", ("a", 2): "stop"}

    result = _stop_metrics(rows, targets)
    assert result["decision_accuracy"] == 2 / 3
    assert result["first_stop_early"] == 1
    assert result["first_stop_exact"] == 0


def test_explicit_stop_evaluation_separates_timing_and_outcome():
    rows = [
        {"episode_id": "a", "decision_index": 0, "expected_decision": "continue",
         "predicted_decision": "stop", "expected_outcome": None, "predicted_outcome": "fail",
         "stop_reason": "further_evidence_required"},
        {"episode_id": "a", "decision_index": 1, "expected_decision": "stop",
         "predicted_decision": "stop", "expected_outcome": "fail", "predicted_outcome": "fail",
         "stop_reason": "evidence_sufficient"},
        {"episode_id": "b", "decision_index": 0, "expected_decision": "stop",
         "predicted_decision": "continue", "expected_outcome": "uncertain", "predicted_outcome": None,
         "stop_reason": "budget_exhausted_uncertain"},
    ]
    result = _summarize(rows)
    assert result["decision_accuracy"] == 1 / 3
    assert result["premature_stop"] == 1 and result["missed_stop"] == 1
    assert result["timing_eligible_episodes"] == 2
    assert result["first_stop_timing"] == {"early": 1, "missing": 1}
    assert result["budget_exhausted"] == {"n": 1, "correct_uncertain_stop": 0}
    assert _parse('{"stop_decision":{"decision":"stop","outcome":"fail","reason":"x"}}') == ("stop", "fail")
    with pytest.raises(ValueError, match="Continue must have null outcome"):
        _parse('{"stop_decision":{"decision":"continue","outcome":"fail"}}')


def test_stop_stage_subset_without_terminal_has_no_timing_denominator():
    result = _summarize([{
        "episode_id": "a", "decision_index": 0, "expected_decision": "continue",
        "predicted_decision": "continue", "expected_outcome": None, "predicted_outcome": None,
        "stop_reason": "further_evidence_required",
    }])
    assert result["timing_eligible_episodes"] == 0
    assert result["first_stop_timing"] == {}


def _audit_row(uid, assistant, *, step=0, terminal=True):
    payload = {"remaining": 1, "history": [{"step": 0}], "training_objective": "first_failure"}
    return {
        "id": uid, "episode_id": uid.split(":")[0], "decision_index": step,
        "terminal": terminal, "evidence": {"covered_label_indices": []},
        "images": ["reference.png"], "image_sha256": ["same"],
        "messages": [
            {"role": "user", "content": "protocol\n" + json.dumps(payload)},
            {"role": "assistant", "content": assistant},
        ],
    }


def _write_integrity_fixture(root, episodes, stopping):
    for split in ("train", "validation"):
        rows = episodes if split == "train" else []
        stops = stopping if split == "train" else []
        (root / f"{split}.episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (root / f"{split}.stopping.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in stops), encoding="utf-8")


def test_supervision_integrity_reports_leakage_and_equal_prefix_conflicts(tmp_path):
    left = _audit_row("a:verdict", json.dumps({"verdict": {
        "status": "uncertain", "findings": [], "summary": "Hidden defect not observed.",
        "limitations": []}}))
    right = _audit_row("b:verdict", json.dumps({"verdict": {
        "status": "uncertain", "findings": [], "summary": "Current evidence is insufficient.",
        "limitations": []}}))
    stop = lambda uid: {**_audit_row(uid, json.dumps({"stop_decision": {
        "decision": "stop", "outcome": "uncertain", "reason": "Budget exhausted."}})),
                        "stop_target": "stop", "stop_outcome": "uncertain"}
    _write_integrity_fixture(tmp_path, [left, right], [stop("a:stop"), stop("b:stop")])

    report = _integrity(tmp_path)
    assert report["assistant_leak_examples"] == [{"id": "a:verdict", "terms": ["hidden defect"]}]
    assert len(report["exact_public_prefix_conflicts"]) == 1


def test_supervision_integrity_rejects_future_evidence_step(tmp_path):
    row = _audit_row("a:verdict", json.dumps({"verdict": {
        "status": "fail", "findings": [{"evidence_steps": [1]}],
        "summary": "Observed failure.", "limitations": []}}))
    stop = {**_audit_row("a:stop", json.dumps({"stop_decision": {
        "decision": "stop", "outcome": "uncertain", "reason": "Budget exhausted."}})),
            "stop_target": "stop", "stop_outcome": "uncertain"}
    _write_integrity_fixture(tmp_path, [row], [stop])

    with pytest.raises(ValueError, match="future or unknown step"):
        _integrity(tmp_path)
