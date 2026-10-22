"""Issue #9: code-level shortcuts of joint_axis / direction_flip and of the code_read condition."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from turnitover.assets.emit import emit_program
from turnitover.benchmark.code_shortcuts import audit_dataset, joint_features, loo_predictions
from turnitover.benchmark.manifest import candidate_program_source
from turnitover.benchmark.protocol import (CONDITIONS_V2, MOTION_EVIDENCE_V2, InvalidOutput, condition_inputs,
                                           finding_consistent, parse_prediction, protocol_document_v2)
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.core.spec import JointType
from turnitover.corruptions import get_corruption
from turnitover.corruptions.kinematics import coordinate_axis

ROOT = Path(__file__).resolve().parents[2]


def _joint_pose(joint, value):
    """Pivot pose after ``set(value)`` exactly as the emitted program computes it (rotation matrix, translation)."""
    axis = np.asarray(joint.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    if joint.type is JointType.REVOLUTE:
        return Rotation.from_rotvec(axis * joint.sign * value).as_matrix(), np.asarray(joint.anchor)
    return np.eye(3), np.asarray(joint.anchor) + axis * joint.sign * value


# ---- (a) axis permutation -------------------------------------------------------------------------------------

def test_axis_permutation_swaps_to_another_signed_coordinate_axis(toy_program):
    corruption = get_corruption("kinematics.joint_axis", mode="axis_permutation")
    seen = set()
    for seed in range(200):
        program, label = corruption.apply(toy_program, np.random.default_rng(seed))
        joint_id = label.params["joint"]
        old, new = toy_program.spec.joint(joint_id), program.spec.joint(joint_id)
        source, target = coordinate_axis(old.axis), coordinate_axis(new.axis)
        assert target is not None and target[0] != source[0]
        assert (new.sign, new.anchor, new.limits, new.type) == (old.sign, old.anchor, old.limits, old.type)
        assert label.params["mode"] == "axis_permutation" and label.params["angle_deg"] == 90.0
        assert (label.params["from_index"], label.params["to_index"], label.params["to_sign"]) == (
            source[0], target[0], target[1])
        assert all(math.copysign(1.0, x) > 0 for x in new.axis if x == 0.0)  # no "-0.0" literal in the code
        assert program.spec.parts == toy_program.spec.parts
        assert [j for j in program.spec.joints if j != toy_program.spec.joint(j.id)] == [new]
        seen.add((source[0], target))
    # both signs and both other axes occur for the same source axis, drawn from the explicit generator
    assert {(1, (0, 1)), (1, (0, -1)), (1, (2, 1)), (1, (2, -1))} <= seen
    first = corruption.apply(toy_program, np.random.default_rng(5))
    assert first[0].source == corruption.apply(toy_program, np.random.default_rng(5))[0].source


def test_axis_permutation_skips_joints_whose_axis_is_not_a_coordinate_axis(toy_spec):
    tilted = toy_spec.replace_joint("door", axis=(0.0, math.sqrt(0.5), math.sqrt(0.5)))
    only_tilted = type(tilted)(tilted.asset_id, tilted.category, tilted.parts,
                               tuple(j for j in tilted.joints if j.id == "door"), tilted.materials)
    corruption = get_corruption("kinematics.joint_axis", mode="axis_permutation")
    assert not corruption.applicable(ObjectProgram.from_spec(only_tilted))
    with pytest.raises(ValueError):
        corruption.apply(ObjectProgram.from_spec(only_tilted), np.random.default_rng(0))
    mixed = ObjectProgram.from_spec(tilted)  # drawers still qualify; the tilted door is never picked
    assert corruption.applicable(mixed)
    assert all(corruption.apply(mixed, np.random.default_rng(s))[1].params["joint"] != "door" for s in range(50))


def test_random_angle_stays_a_separate_stratum_and_leaves_the_coordinate_axes(toy_program):
    corruption = get_corruption("kinematics.joint_axis")
    assert corruption.mode == "axis_random_angle"
    _, label = corruption.apply(toy_program, np.random.default_rng(1))
    assert label.params["mode"] == "axis_random_angle"
    assert all(coordinate_axis(corruption.apply(toy_program, np.random.default_rng(s))[1].label["axis"]) is None
               for s in range(20))
    with pytest.raises(ValueError):
        get_corruption("kinematics.joint_axis", mode="nonsense")


# ---- (b) axis negation == sign negation -----------------------------------------------------------------------

def test_direction_flip_negates_axis_and_keeps_sign(toy_program):
    corruption = get_corruption("kinematics.direction_flip")
    for seed in range(20):
        program, label = corruption.apply(toy_program, np.random.default_rng(seed))
        old, new = toy_program.spec.joint(label.params["joint"]), program.spec.joint(label.params["joint"])
        assert new.sign == old.sign
        assert new.axis == tuple(-x for x in old.axis) and (new.anchor, new.limits) == (old.anchor, old.limits)
        assert label.params["mode"] == "axis_negation"
        assert all(math.copysign(1.0, x) > 0 for x in new.axis if x == 0.0)


@pytest.mark.parametrize("joint_id", ["door", "drawer_0"])  # one revolute, one prismatic joint
def test_axis_negation_moves_exactly_like_sign_negation(toy_spec, joint_id):
    joint = toy_spec.joint(joint_id)
    axis_negated = toy_spec.replace_joint(joint_id, axis=tuple(0.0 - x for x in joint.axis)).joint(joint_id)
    sign_negated = toy_spec.replace_joint(joint_id, sign=-joint.sign).joint(joint_id)
    assert axis_negated != sign_negated  # different code ...
    for value in np.linspace(joint.limits[0], joint.limits[1], 9):
        for a, b in zip(_joint_pose(axis_negated, value), _joint_pose(sign_negated, value)):
            np.testing.assert_allclose(a, b, atol=1e-12)  # ... identical motion
    original = _joint_pose(joint, joint.limits[1])
    assert not np.allclose(original[0], _joint_pose(axis_negated, joint.limits[1])[0]) or not np.allclose(
        original[1], _joint_pose(axis_negated, joint.limits[1])[1])  # and it really differs from the clean motion


# ---- (c) protocol v2 condition boundary -----------------------------------------------------------------------

def _row():
    roles = ("reference_rest", "reference_limit", "candidate_rest", "candidate_limit")
    return {
        "sample_id": "s1", "prompt": "p", "context": {"joint_ids": ["j"], "part_ids": ["l"]},
        "images": [{"index": i, "role": r, "path": f"img/{r}.png", "sha256": "x"} for i, r in enumerate(roles)],
        "code": [{"role": r, "path": f"code/{r}"} for r in ("reference_spec", "candidate_spec", "candidate_program")],
        "status": "fail", "defects": [{"defect_id": "kinematics.joint_axis"}], "review": {"status": "done"},
        "frozen_state": {"joint_id": "j"}, "camera": {}, "split": "train", "difficulty": "easy", "asset_id": "a"}


def test_code_read_inputs_hold_candidate_code_and_reference_images_only():
    visible = condition_inputs(_row(), "code_read", CONDITIONS_V2)
    assert set(visible) == {"sample_id", "prompt", "context", "images", "code"}
    assert [i["role"] for i in visible["images"]] == ["reference_rest", "reference_limit"]
    assert [c["role"] for c in visible["code"]] == ["candidate_program"]
    text = json.dumps(visible)
    for hidden in ("reference_spec", "candidate_spec", "candidate_rest", "candidate_limit",
                   "defects", "review", "status", "frozen_state", "asset_id", "difficulty", "split"):
        assert hidden not in text
    # the evidence index is defined by the condition, not by the manifest order
    assert [e["role"] for e in CONDITIONS_V2["code_read"]["evidence"]] == [
        "reference_rest", "reference_limit", "candidate_program"]
    assert CONDITIONS_V2["code_read"]["n_evidence"] == 3


def test_every_v2_condition_reads_only_public_fields_in_evidence_order():
    public = {"sample_id", "prompt", "context", "images", "code"}
    row = _row()
    for name, condition in CONDITIONS_V2.items():
        visible = condition_inputs(row, name, CONDITIONS_V2)
        assert set(visible) <= public and set(condition["fields"]) <= public, name
        listed = [(e["field"], e["role"]) for e in condition["evidence"]]
        assert condition["n_evidence"] == len(listed), name
        flattened = [(f, e["role"]) for f in ("images", "code") for e in visible.get(f, [])]
        assert sorted(flattened) == sorted(listed), name
    assert [c["role"] for c in condition_inputs(row, "code_diff", CONDITIONS_V2)["code"]] == [
        "reference_spec", "candidate_spec"]
    assert "images" not in condition_inputs(row, "code_diff", CONDITIONS_V2)
    assert "diagnostic" in CONDITIONS_V2["code_diff"]["note"].lower()


def test_v2_evidence_indices_are_validated_and_motion_pair_is_reference_limit_plus_code():
    ok = {"case_id": "a", "decision": "fail", "findings": [
        {"defect_id": "kinematics.joint_axis", "joint_id": "j", "parts": ["l"], "evidence": [1, 2]}]}
    kwargs = dict(joint_ids={"j"}, part_ids={"l"}, conditions=CONDITIONS_V2)
    prediction = parse_prediction(ok, condition="code_read", **kwargs)
    assert finding_consistent(prediction.findings[0], "code_read", MOTION_EVIDENCE_V2)
    cites_images_only = parse_prediction({**ok, "findings": [{**ok["findings"][0], "evidence": [0, 1]}]},
                                         condition="code_read", **kwargs)
    assert not finding_consistent(cites_images_only.findings[0], "code_read", MOTION_EVIDENCE_V2)
    with pytest.raises(InvalidOutput):
        parse_prediction({**ok, "findings": [{**ok["findings"][0], "evidence": [3]}]}, condition="code_read", **kwargs)


def test_protocol_v2_file_is_the_code_and_candidate_program_is_deterministic(toy_spec, tmp_path):
    expected = json.dumps(protocol_document_v2(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    assert (ROOT / "benchmark/v2/protocol.json").read_text(encoding="utf-8") == expected
    assert not (ROOT / "benchmark/v2/FREEZE.json").exists()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(to_dict(toy_spec)), encoding="utf-8")
    source, digest = candidate_program_source(path)
    assert source == emit_program(toy_spec) and candidate_program_source(path) == (source, digest)


# ---- audit rules ------------------------------------------------------------------------------------------------

def _sample(label, stratum, axis, sign=1.0, anchor=(0.0, 0.0, 0.0)):
    return {"label": label, "stratum": stratum, "joints": [
        {"id": "j", "axis": axis, "sign": sign, "anchor": anchor, "limits": [0, 1], "part_origin": [0, 0, 0]}]}


def test_audit_flags_known_shortcuts_and_not_matched_distributions():
    unit = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0))
    clean = [_sample("clean", None, unit[i % 4]) for i in range(40)]
    tilted = [_sample("d", "axis_random_angle", (0.6, 0.8, 0.0)) for _ in range(40)]
    negative_sign = [_sample("d", "sign_negation", unit[i % 4], sign=-1.0) for i in range(40)]
    matched = [_sample("d", "axis_negation", tuple(0.0 - x for x in unit[(i + 2) % 4])) for i in range(40)]
    result = audit_dataset(clean + tilted + negative_sign + matched)
    assert result["d/axis_random_angle"]["rules"]["axis_off_coordinate"]["balanced_accuracy"] == 1.0
    assert result["d/sign_negation"]["rules"]["sign_negative"]["balanced_accuracy"] == 1.0
    assert result["d/sign_negation"]["above_chance"] == ["sign_negative"]
    flip = result["d/axis_negation"]
    assert flip["rules"]["axis_off_coordinate"]["balanced_accuracy"] == 0.5 and flip["rules"]["sign_negative"][
        "balanced_accuracy"] == 0.5
    assert flip["rules"]["axis_dominant_negative"]["balanced_accuracy"] == 0.5


def test_leave_one_out_prior_cannot_use_the_sample_it_predicts():
    truth = np.array([False, False, True, True])
    unique = loo_predictions(["a", "b", "c", "d"], truth)  # every key is unseen once the sample is left out
    assert not unique.any()
    informative = loo_predictions(["a", "a", "b", "b"] * 5, np.array([False, False, True, True] * 5))
    assert informative.tolist() == [False, False, True, True] * 5
    assert [j["id"] for j in joint_features(_toy_spec_dict())] == ["drawer_0", "drawer_1", "door"]


def _toy_spec_dict():
    from turnitover.assets.toy import toy_cabinet
    return to_dict(toy_cabinet())
