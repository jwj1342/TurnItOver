import json

from turnitover.verifier.contracts import Finding, Verdict
from turnitover.verifier.modular import (
    annotation_witness_indices,
    build_single_reference_record,
    gate_findings,
)
from scripts.audit_modular_verifier import _aggregate


def _finding(defect, steps, joint=None):
    return Finding(defect, ("part",), joint, .5, .8, tuple(steps), "description", "fix")


def test_build_single_reference_record_preserves_protocol_and_replaces_only_prefix():
    payload = {"context": {"task": "t"}, "remaining": 6, "history": [],
               "image_order": ["old.png"], "taxonomy_ids": ["x"]}
    template = {"images": ["old.png"], "messages": [
        {"role": "user", "content": "protocol\n" + json.dumps(payload)},
        {"role": "assistant", "content": "target"},
    ]}
    history = [{"step": 0, "action": {"type": "request_view", "view_id": "front"},
                "image_ref": "private.png", "payload": {"camera": {}}, "reference_index": 0}]
    record = build_single_reference_record(template, history, 5)
    public = json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])
    assert record["messages"][0]["content"].startswith("protocol\n")
    assert public["context"] == {"task": "t"}
    assert public["remaining"] == 5
    assert public["history"][0]["image_ref"] == "observation_0.png"
    assert "reference_index" not in public["history"][0]
    assert record["images"] == ["reference.png", "observation_0.png"]


def test_gate_rejects_static_axis_but_accepts_same_joint_actuation():
    history = [
        {"step": 0, "action": {"type": "request_view", "view_id": "front"}},
        {"step": 1, "action": {"type": "actuate_joint", "joint_id": "joint_0", "detent": "limit"}},
    ]
    evidence = [{"state": "sufficient"}, {"state": "sufficient"}]
    static = Verdict("fail", .8, (_finding("kinematics.joint_axis", [0], "joint_0"),), "s", ())
    accepted, rejected = gate_findings(static, history, evidence, require_aligned_visual=True)
    assert accepted == []
    assert rejected[0]["reason"] == "axis_claim_lacks_cited_same_joint_actuation"
    dynamic = Verdict("fail", .8, (_finding("kinematics.joint_axis", [1], "joint_0"),), "s", ())
    accepted, rejected = gate_findings(dynamic, history, evidence, require_aligned_visual=True)
    assert len(accepted) == 1
    assert rejected == []


def test_gate_requires_v5_mismatch_for_modular_offset():
    history = [{"step": 0, "action": {"type": "request_view", "view_id": "front"}}]
    verdict = Verdict("fail", .8, (_finding("structure.part_offset", [0]),), "s", ())
    assert gate_findings(verdict, history, [{"state": "identical"}],
                         require_aligned_visual=True)[0] == []
    assert len(gate_findings(verdict, history, [{"state": "sufficient"}],
                             require_aligned_visual=True)[0]) == 1


def test_aligned_reference_states_map_to_v4_grid():
    annotation = {"grid": [
        {"view": "front", "joint": None, "detent": None},
        {"view": "front", "joint": "joint_0", "detent": "limit"},
    ]}
    states = [
        {"view": "front", "joint": "joint_0", "detent": "rest"},
        {"view": "front", "joint": "joint_0", "detent": "limit"},
        {"view": "left", "joint": "joint_0", "detent": "limit"},
    ]
    history = [{"reference_index": 0}, {"reference_index": 1}, {"reference_index": 2}]
    assert annotation_witness_indices(annotation, history, states) == [0, 1, None]


def test_episode_aggregate_separates_type_from_part_localization():
    row = {
        "asset_id": "asset", "final_status": "fail", "gold_stratum": "offset_only",
        "expected_visible_findings": [["structure.part_offset", None, ["right_part"]]],
        "accepted_findings": [{"defect_id": "structure.part_offset", "joint_id": None,
                               "parts": ["wrong_part"]}],
        "unsupported_findings": 1, "calls": [], "stop_timing": "exact",
        "evidence_sufficient_at_fail": False, "observation_count": 1,
        "v4_unmapped_observations": 0,
    }
    metrics = _aggregate([row])
    assert metrics["defect_type_micro"]["f1"] == 1
    assert metrics["part_localization_micro"]["f1"] == 0
    assert metrics["status_accuracy"] == 1
