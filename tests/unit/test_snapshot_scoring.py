"""Synthetic scoring boundaries for frozen paper metrics."""
import copy
import json

import pytest

from turnitover.oracle.snapshot_scoring import score_snapshot


OFFSET = "structure.part_offset"
AXIS = "kinematics.joint_axis"


def inputs(*, paired=True, labels=None, pairs=None):
    finding = {"defect_id": OFFSET, "parts": ["door"], "joint_id": None,
               "severity": 1., "confidence": 1., "evidence_steps": [0, 1],
               "description": "Offset", "suggested_fix": "Align"}
    target = {"verdict": {"status": "fail", "confidence": 1., "findings": [finding],
                          "summary": "Offset", "limitations": []}}
    payload = {"context": {"task": "diagnose", "parts": ["door", "frame"],
                           "joints": ["hinge"], "views": [], "runtime_properties": [],
                           "action_types": [], "max_triangles": 100, "max_draw_calls": 10},
               "history": [{"step": step} for step in range(3)]}
    if paired:
        payload["observation_format"] = "reference_candidate_pairs_v1"
    record = {"id": "a:verdict", "episode_id": "a", "difficulty": "easy",
              "data_identity": {"source_identity_sha256": "source-a"},
              "is_clean": False,
              "messages": [{"role": "user", "content": "Protocol\n" + json.dumps(payload)},
                           {"role": "assistant", "content": json.dumps(target)}],
              "evidence": {"pair_label_indices": pairs if pairs is not None else [[0], [0], []],
                           "covered_label_indices": [0]}}
    annotation = {"id": "a", "labels": labels if labels is not None else [
        {"defect_id": OFFSET, "parts": ["door"], "params": {}},
        {"defect_id": AXIS, "parts": ["door"], "params": {"joint": "hinge"}}]}
    return [record], [annotation], [{"id": record["id"], "raw": json.dumps(target)}]


def score(args):
    return score_snapshot(*args, [OFFSET, AXIS])


def test_injected_unexposed_labels_are_false_negatives_without_changing_targets():
    report = score(inputs())
    assert report["target_metrics"]["overall"]["defect_per_class"][AXIS]["fn"] == 0
    injected = report["paper_metrics"]["all_injected_metrics"]["overall"]
    assert injected["defect_per_class"][AXIS]["fn"] == 1
    assert injected["part_localization_micro"]["recall"] == .5
    assert "status_accuracy" not in injected
    assert report["paper_metrics"]["exposure_any_rate"] == 1
    assert report["paper_metrics"]["exposure_all_rate"] == 0
    assert report["paper_metrics"]["strict_evidence_support_rate"] == 1


def test_exact_citations_reject_a_supported_subset_and_superset():
    for steps in ([0], [0, 1, 2]):
        args = inputs()
        raw = json.loads(args[2][0]["raw"])
        raw["verdict"]["findings"][0]["evidence_steps"] = steps
        args[2][0]["raw"] = json.dumps(raw)
        report = score(args)
        assert report["paper_metrics"]["strict_evidence_support_rate"] == 0
        if steps == [0]:
            assert report["evidence_support_rate"] == 1  # Legacy subset convention retained.


def test_duplicate_signatures_union_support():
    labels = [{"defect_id": OFFSET, "parts": ["door"], "params": {}}] * 2
    args = inputs(labels=labels, pairs=[[0], [1], []])
    args[0][0]["evidence"]["covered_label_indices"] = [0, 1]
    assert score(args)["paper_metrics"]["strict_evidence_support_rate"] == 1


def test_unexposed_is_type_based_even_when_full_localization_is_wrong():
    args = inputs()
    raw = json.loads(args[2][0]["raw"])
    raw["verdict"]["findings"][0]["parts"] = ["frame"]
    args[2][0]["raw"] = json.dumps(raw)
    paper = score(args)["paper_metrics"]
    assert paper["strict_evidence_support_rate"] == 0
    assert paper["unexposed_finding_rate"] == 0
    raw["verdict"]["findings"][0].update(defect_id=AXIS, joint_id="hinge")
    args[2][0]["raw"] = json.dumps(raw)
    assert score(args)["paper_metrics"]["unexposed_finding_rate"] == 1


def test_clean_empty_output_has_null_denominators():
    args = inputs(labels=[], pairs=[[], [], []])
    record = args[0][0]
    record["is_clean"] = True
    record["evidence"]["covered_label_indices"] = []
    target = json.loads(record["messages"][1]["content"])
    target["verdict"].update(status="uncertain", findings=[], limitations=["Limited views"])
    record["messages"][1]["content"] = json.dumps(target)
    args[2][0]["raw"] = json.dumps(target)
    paper = score(args)["paper_metrics"]
    assert paper["predicted_findings"] == 0
    assert paper["strict_evidence_support_rate"] is None
    assert paper["unexposed_finding_rate"] is None
    assert paper["exposure_any_rate"] is None
    assert paper["exposure_all_rate"] is None


@pytest.mark.parametrize("mode", ["duplicate", "missing", "extra"])
def test_prediction_coverage_must_be_unique_and_exact(mode):
    args = inputs()
    if mode == "duplicate":
        args[2].append(copy.deepcopy(args[2][0]))
    elif mode == "missing":
        args[2].clear()
    else:
        args[2].append({"id": "extra", "raw": "bad"})
    with pytest.raises(ValueError):
        score(args)


def test_unknown_prediction_type_invalidates_output_but_keeps_case():
    args = inputs()
    raw = json.loads(args[2][0]["raw"])
    raw["verdict"]["findings"][0]["defect_id"] = "invented.defect"
    args[2][0]["raw"] = json.dumps(raw)
    report = score(args)
    assert report["invalid_outputs"] == 1
    assert report["paper_metrics"]["all_injected_metrics"]["overall"]["defect_per_class"][OFFSET]["fn"] == 1


def test_missing_full_exposure_mapping_is_explicitly_unavailable():
    report = score(inputs(paired=False))
    paper = report["paper_metrics"]
    assert paper["exposure_any_rate"] is None
    assert paper["unexposed_finding_rate"] is None
    assert paper["exposure_unavailable_reason"]
    assert paper["strict_evidence_support_rate"] == 1
    assert "target_expected_evidence_not_complete_exposure_audit" in paper["strict_evidence_support_source"]


def test_corrupt_paired_mapping_is_rejected():
    args = inputs(pairs=[[1], [1], []])
    with pytest.raises(ValueError, match="coverage"):
        score(args)


def test_duplicate_predicted_findings_remain_in_paper_denominator():
    args = inputs()
    raw = json.loads(args[2][0]["raw"])
    wrong = copy.deepcopy(raw["verdict"]["findings"][0])
    wrong["evidence_steps"] = [0]
    raw["verdict"]["findings"].append(wrong)
    args[2][0]["raw"] = json.dumps(raw)
    paper = score(args)["paper_metrics"]
    assert paper["predicted_findings"] == 2
    assert paper["strict_evidence_support_rate"] == .5
