import json

import pytest

from turnitover.oracle.episodes import episode_examples
from turnitover.oracle.stopping import stopping_examples


CONTEXT = {
    "task": "verify", "parts": ["door"], "joints": ["hinge"],
    "views": ["front"], "runtime_properties": [],
    "action_types": ["request_view", "actuate_joint"],
    "max_triangles": 5000, "max_draw_calls": 32,
}
HISTORY = [
    {"step": 0, "action": {"type": "request_view", "view_id": "front"},
     "image_ref": "view.png", "payload": {}},
    {"step": 1, "action": {"type": "actuate_joint", "joint_id": "hinge", "detent": "limit"},
     "image_ref": "motion.png", "payload": {}},
]
LABELS = [
    {"defect_id": "structure.part_offset", "parts": ["door"], "severity": 0.5,
     "params": {"vector": [0.1, 0.0, 0.0]}},
    {"defect_id": "kinematics.joint_axis", "parts": ["door"], "severity": 0.75,
     "params": {"joint": "hinge"}},
]


def _terminal(exposures, labels=LABELS, objective="first_failure"):
    rows = episode_examples(
        "case", CONTEXT, "reference.png", HISTORY, [0, 1], exposures, labels, budget=2,
        objective=objective)
    assert [row["terminal"] for row in rows] == [False] * (len(rows) - 1) + [True]
    assert rows[-1]["decision_index"] == len(rows) - 1
    assert all(row["episode_id"] == "case" for row in rows)
    assert all(row["training_objective"] == objective for row in rows)
    assert all(json.loads(row["messages"][0]["content"].rsplit("\n", 1)[-1])["training_objective"] == objective
               for row in rows)
    return rows[-1], json.loads(rows[-1]["messages"][1]["content"])["verdict"]


def test_combined_episode_has_actions_evidence_diagnosis_and_one_stop_target():
    row, verdict = _terminal([[0, 1], []])

    assert row["decision_index"] == 1
    assert row["termination_reason"] == "evidence_sufficient"
    assert row["termination_target"] == "verdict"
    assert row["evidence"]["status"] == "complete"
    assert verdict["status"] == "fail"
    assert [finding["evidence_steps"] for finding in verdict["findings"]] == [[0], [0]]
    assert "vector" not in row["messages"][0]["content"]


@pytest.mark.parametrize(("exposures", "evidence_status", "verdict_status"), [
    ([[0], []], "partial", "fail"),
    ([[], []], "none", "uncertain"),
])
def test_incomplete_evidence_never_invents_hidden_findings(
        exposures, evidence_status, verdict_status):
    row, verdict = _terminal(exposures)

    assert row["evidence"]["status"] == evidence_status
    assert verdict["status"] == verdict_status
    assert len(verdict["findings"]) == (1 if evidence_status == "partial" else 0)
    assert row["termination_reason"] == ("evidence_sufficient" if verdict_status == "fail" else "budget_exhausted")


def test_clean_episode_never_uses_hidden_label_absence_as_pass_evidence():
    row, verdict = _terminal([[], []], labels=[])

    assert row["evidence"]["status"] == "none"
    assert row["termination_reason"] == "budget_exhausted"
    assert verdict["status"] == "uncertain" and not verdict["findings"]
    assert verdict["limitations"]


def test_no_evidence_target_is_identical_for_clean_and_hidden_defect_cases():
    _, clean = _terminal([[], []], labels=[])
    _, hidden = _terminal([[], []], labels=LABELS)

    assert clean == hidden
    text = json.dumps(clean).lower()
    assert all(term not in text for term in ("inject", "private", "hidden", "unexposed"))


def test_failure_stops_at_first_exposed_step_instead_of_exhausting_budget():
    row, verdict = _terminal([[], [0]])

    assert row["decision_index"] == 2
    assert row["termination_reason"] == "evidence_sufficient"
    assert verdict["status"] == "fail"
    assert verdict["findings"][0]["evidence_steps"] == [1]


def test_comprehensive_objective_keeps_budgeted_history_and_all_exposed_findings():
    row, verdict = _terminal([[0], [1]], objective="comprehensive_within_budget")

    assert row["decision_index"] == 2
    assert row["termination_reason"] == "budget_exhausted"
    assert row["evidence"]["status"] == "complete"
    assert [finding["evidence_steps"] for finding in verdict["findings"]] == [[0], [1]]


def test_unknown_training_objective_is_rejected():
    with pytest.raises(ValueError, match="Unknown Episode training objective"):
        episode_examples("case", CONTEXT, "reference.png", HISTORY, [0, 1], [[], []], [], 2, "unknown")


def test_terminal_target_requires_an_observation():
    with pytest.raises(ValueError, match="at least one observation"):
        episode_examples("case", CONTEXT, "reference.png", [], [], [], [], budget=2)


def _stop_targets(exposures, labels=LABELS, objective="first_failure"):
    return list(stopping_examples(
        "case", CONTEXT, "reference.png", HISTORY, [0, 1], exposures, labels,
        budget=2, objective=objective))


def test_first_failure_stop_targets_include_continue_then_evidence_stop():
    rows = _stop_targets([[], [0]])

    assert [row["stop_target"] for row in rows] == ["continue", "continue", "stop"]
    assert rows[-1]["stop_outcome"] == "fail"
    assert rows[-1]["stop_reason"] == "evidence_sufficient"


def test_budget_exhausted_uncertain_is_separate_from_evidence_stop():
    rows = _stop_targets([[], []])

    assert [row["stop_target"] for row in rows] == ["continue", "continue", "stop"]
    assert rows[-1]["stop_outcome"] == "uncertain"
    assert rows[-1]["stop_reason"] == "budget_exhausted_uncertain"


def test_comprehensive_policy_continues_after_first_evidence_until_final_prefix():
    rows = _stop_targets([[0], [1]], objective="comprehensive_within_budget")

    assert [row["stop_target"] for row in rows] == ["continue", "continue", "stop"]
    assert rows[-1]["stop_outcome"] == "fail"


def test_no_evidence_stop_supervision_is_private_gold_invariant():
    clean = _stop_targets([[], []], labels=[])
    hidden = _stop_targets([[], []], labels=LABELS)

    assert [row["messages"][1] for row in clean] == [row["messages"][1] for row in hidden]
