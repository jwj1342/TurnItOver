import json

import pytest

from turnitover.oracle.episodes import episode_examples


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


def _terminal(exposures, labels=LABELS):
    rows = episode_examples(
        "case", CONTEXT, "reference.png", HISTORY, [0, 1], exposures, labels, budget=2)
    assert [row["terminal"] for row in rows] == [False, False, True]
    assert all(row["episode_id"] == "case" for row in rows)
    return rows[-1], json.loads(rows[-1]["messages"][1]["content"])["verdict"]


def test_combined_episode_has_actions_evidence_diagnosis_and_one_stop_target():
    row, verdict = _terminal([[0], [1]])

    assert row["termination_target"] == "verdict"
    assert row["evidence"]["status"] == "complete"
    assert verdict["status"] == "fail"
    assert [finding["evidence_steps"] for finding in verdict["findings"]] == [[0], [1]]
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


def test_clean_episode_emits_scoped_pass():
    row, verdict = _terminal([[], []], labels=[])

    assert row["evidence"]["status"] == "complete"
    assert verdict["status"] == "pass" and not verdict["findings"]
    assert verdict["limitations"]


def test_terminal_target_requires_an_observation():
    with pytest.raises(ValueError, match="at least one observation"):
        episode_examples("case", CONTEXT, "reference.png", [], [], [], [], budget=2)
