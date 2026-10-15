import hashlib
import json
from types import SimpleNamespace

import pytest

from turnitover.core.actions import detent_value
from turnitover.teacher.audit import audit_episode
from turnitover.teacher.episode import run_teacher_episode
from turnitover.teacher.export import export_episodes
from turnitover.teacher.policy import TeacherPolicy
from turnitover.teacher.providers import MockTeacher
from turnitover.teacher.witnesses import (clamp, fixed_protocol_actions, row_sha256, witness_state)
from turnitover.verifier.contracts import Context


IDENTITY = {"source_id": "source", "asset_id": "asset", "split_group": "group",
            "candidate_sha": "c" * 16, "reference_sha": "r" * 16, "formal_training": True,
            "split": "train", "source_identity_sha256": "a" * 64, "source_content_sha256": "b" * 64,
            "experiment_mode": "active", "reference_protocol": "front_motion_v1"}


class Session:
    """Renders pixels as a deterministic function of the actual (clamped) state."""

    def __init__(self, limits):
        self.limits, self.view = limits, "front"
        self.values = {joint: clamp(0.0, bounds) for joint, bounds in limits.items()}

    def png(self):
        return json.dumps([self.view, self.values], sort_keys=True).encode()

    def request_view(self, view):
        self.view = view
        return SimpleNamespace(png=self.png(), render_ms=1, camera={"view": view})

    def actuate(self, joint, detent):
        self.values[joint] = clamp(detent_value(detent, self.limits[joint]), self.limits[joint])
        return SimpleNamespace(png=self.png(), render_ms=1, value=self.values[joint])

    def query_runtime(self, prop):
        return {"joints": dict(self.values)}


def context():
    return Context("inspect", ("base", "door"), ("hinge",), ("front", "side"), ("joint_state",),
                   ("request_view", "actuate_joint", "query_runtime"), 100, 10)


def build_case(tmp_path, limits, grid, exposures, labels, *, reference_rows=None):
    """Frozen manifest whose pixels follow the same deterministic Session rendering."""
    folder = tmp_path / "frozen"
    folder.mkdir(exist_ok=True)
    witnesses = []
    for index, row in enumerate(grid):
        state = witness_state(row, limits)
        png = json.dumps([row["view"], state], sort_keys=True).encode()
        witnesses.append({"index": index, "key": f"{row['view']}:{row.get('joint') or 'rest'}",
                          "view": row["view"], "joint": row.get("joint"), "detent": row.get("detent"),
                          "state": state, "sha256": hashlib.sha256(png).hexdigest(),
                          "exposures": exposures[index]})
    references = []
    for position, row in enumerate(reference_rows if reference_rows is not None else grid):
        path = folder / f"reference-{position:02d}.png"
        path.write_bytes(f"reference:{position}".encode())
        references.append({"file": path.name, "state": witness_state(row, limits),
                           "label": {"view": row["view"], "joint_id": row.get("joint"),
                                     "detent": row.get("detent") or "rest"},
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    gold = {"id": "case", "source_id": "source", "asset_id": "asset", "split_group": "group",
            "program_sha": "c" * 16, "reference_sha": "r" * 16, "quality": {"status": "accepted"},
            "provenance": {"formal_training": True},
            "target": {"findings": [{"description": "secret gold text", **label} for label in labels]}}
    annotation = {"id": "case", "split": "train", "asset_id": "asset", "split_group": "group",
                  "candidate_sha": "c" * 16, "reference_sha": "r" * 16, "data_identity": {},
                  "labels": labels, "grid": grid, "exposures": exposures, "trajectories": {}}
    manifest = {"id": "case", "split": "train", "asset_id": "asset", "split_group": "group",
                "candidate_sha": "c" * 16, "reference_sha": "r" * 16, "in_frozen_test": False,
                "row_sha256": {"gold": row_sha256(gold), "annotation": row_sha256(annotation)},
                "candidate_joint_limits": {k: list(v) for k, v in limits.items()},
                "witnesses": witnesses, "static_reference_sha256": "unused",
                "reference_protocol": {"name": "front_motion_v1", "states": references}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return gold, annotation, manifest, folder


def run_case(tmp_path, decisions, limits, case, *, initial=(), budget=None, identity=None):
    gold, annotation, manifest, folder = case
    out = tmp_path / "episode"
    out.mkdir()
    refs = []
    for row in manifest["reference_protocol"]["states"]:
        (out / row["file"]).write_bytes((folder / row["file"]).read_bytes())
        refs.append(out / row["file"])
    policy = TeacherPolicy(MockTeacher(decisions), out, refs,
                           reference_states=[row["label"] for row in manifest["reference_protocol"]["states"]])
    episode = run_teacher_episode(Session(limits), policy, context(), budget or len(decisions) - 1 + len(initial),
                                  out, episode_id="case", data_identity=identity or IDENTITY,
                                  initial_actions=initial)
    episode.update(reference_protocol="front_motion_v1",
                   reference_states=[row["label"] for row in manifest["reference_protocol"]["states"]],
                   runtime_joint_limits={k: list(v) for k, v in limits.items()},
                   frozen_witness_manifest={"path": str(folder / "manifest.json"), "sha256": hashlib.sha256(
                       (folder / "manifest.json").read_bytes()).hexdigest()})
    (out / "episode.json").write_text(json.dumps(episode))
    return out, audit_episode(out, gold, annotation, manifest=manifest)


AXIS = {"defect_id": "kinematics.joint_axis", "parts": ["door"], "joint_id": "hinge",
        "params": {"joint": "hinge"}}
GRID = [{"view": "front", "joint": None, "detent": None},
        {"view": "front", "joint": "hinge", "detent": "zero"},
        {"view": "front", "joint": "hinge", "detent": "limit"},
        {"view": "side", "joint": "hinge", "detent": "limit"}]


def actuate(detent):
    return {"action": {"type": "actuate_joint", "joint_id": "hinge", "detent": detent}}


def finding(steps, joint="hinge", parts=("door",)):
    return {"defect_id": "kinematics.joint_axis", "parts": list(parts), "joint_id": joint, "severity": .7,
            "confidence": .8, "evidence_steps": steps, "description": "Door swings off-axis.",
            "suggested_fix": "Correct the hinge axis."}


def verdict(status, findings=()):
    return {"verdict": {"status": status, "confidence": .8, "findings": list(findings),
                        "summary": "Visible comparison.", "limitations": ["Front view only."]}}


def evidence(audit):
    return audit["independent_evidence_audit"]


LIMITS = {"hinge": (-1.0, 1.0)}


def test_correct_axis_fail_is_bound_by_actual_state_and_pixels(tmp_path):
    case = build_case(tmp_path, LIMITS, GRID, [[], [0], [0], [0]], [AXIS])
    _, audit = run_case(tmp_path, [actuate("limit"), verdict("fail", [finding([0])])], LIMITS, case)
    result = evidence(audit)
    assert result["accepted"] and result["diagnosis_outcome"] == "correct_fail"
    assert result["action_audit"][0]["witness_keys"] == ["front:hinge"]
    assert result["action_audit"][0]["new_objective_evidence"] == [0]
    assert result["localization"][0]["joint_correct"] and result["localization"][0]["parts_correct"]


def test_zero_is_lower_limit_and_rest_is_loaded_pose(tmp_path):
    case = build_case(tmp_path, LIMITS, GRID, [[], [0], [0], [0]], [AXIS])
    _, audit = run_case(tmp_path, [actuate("zero"), actuate("rest"), verdict("uncertain")], LIMITS, case)
    rows = evidence(audit)["action_audit"]
    assert rows[0]["witness_indices"] == [1] and rows[0]["state"]["joints"] == {"hinge": -1.0}
    assert rows[1]["witness_indices"] == [0] and rows[1]["state"]["joints"] == {"hinge": 0.0}
    assert evidence(audit)["diagnosis_outcome"] == "uncertain_after_sufficient_evidence"


def test_rest_clamped_to_positive_lower_limit_coincides_with_zero(tmp_path):
    limits = {"hinge": (0.25, 1.0)}
    grid = [GRID[0], GRID[2]]
    case = build_case(tmp_path, limits, grid, [[], [0]], [AXIS])
    _, audit = run_case(tmp_path, [actuate("limit"), actuate("rest"), verdict("uncertain")], limits, case)
    rows = evidence(audit)["action_audit"]
    assert rows[1]["actual_value"] == 0.25 and rows[1]["actuation_consistent"]
    assert rows[1]["witness_indices"] == [0] and rows[1]["bound"]


def test_uncertain_without_comparable_reference_is_trustworthy(tmp_path):
    case = build_case(tmp_path, LIMITS, GRID, [[], [0], [0], [0]], [AXIS], reference_rows=[GRID[0]])
    _, audit = run_case(tmp_path, [actuate("limit"), verdict("uncertain")], LIMITS, case)
    result = evidence(audit)
    assert result["accepted"] and result["diagnosis_outcome"] == "trustworthy_uncertain"
    assert result["objective_exposure_without_visible_reference"] == [0]


def test_fail_without_evidence_and_wrong_localization_are_rejected(tmp_path):
    case = build_case(tmp_path, LIMITS, GRID, [[], [0], [0], [0]], [AXIS])
    _, guess = run_case(tmp_path, [{"action": {"type": "request_view", "view_id": "front"}},
                                   verdict("fail", [finding([0])])], LIMITS, case)
    assert evidence(guess)["diagnosis_outcome"] == "fail_without_sufficient_evidence"
    assert not evidence(guess)["accepted"]

    other = tmp_path / "other"
    other.mkdir()
    case = build_case(other, LIMITS, GRID, [[], [0], [0], [0]], [AXIS])
    _, wrong = run_case(other, [actuate("limit"), verdict("fail", [finding([0], parts=("base",))])],
                        LIMITS, case)
    assert evidence(wrong)["diagnosis_outcome"] == "incorrect_fail"
    assert evidence(wrong)["localization"][0]["joint_correct"]
    assert not evidence(wrong)["localization"][0]["parts_correct"]


def test_state_outside_grid_and_view_without_reference_are_not_credited(tmp_path):
    case = build_case(tmp_path, LIMITS, GRID, [[], [0], [0], [0]], [AXIS], reference_rows=GRID[:3])
    _, audit = run_case(tmp_path, [actuate("limit"), {"action": {"type": "request_view", "view_id": "side"}},
                                   verdict("fail", [finding([1])])], LIMITS, case)
    rows = evidence(audit)["action_audit"]
    assert rows[1]["bound"] and not rows[1]["reference_comparable"]
    assert "finding_lacks_cited_counterfactual_evidence:kinematics.joint_axis" in evidence(audit)[
        "rejection_reasons"]


@pytest.mark.parametrize("mutate, reason", [
    (lambda gold, ann, identity: identity.update(split="test"), "non_training_or_inconsistent_split"),
    (lambda gold, ann, identity: gold.update(program_sha="d" * 16), "gold_row_sha_differs_from_frozen_manifest"),
    (lambda gold, ann, identity: ann.update(labels=[]), "annotation_row_sha_differs_from_frozen_manifest"),
])
def test_identity_split_and_row_sha_are_enforced(tmp_path, mutate, reason):
    gold, annotation, manifest, folder = build_case(tmp_path, LIMITS, GRID, [[], [0], [0], [0]], [AXIS])
    identity = dict(IDENTITY)
    mutate(gold, annotation, identity)
    _, audit = run_case(tmp_path, [actuate("limit"), verdict("fail", [finding([0])])], LIMITS,
                        (gold, annotation, manifest, folder), identity=identity)
    assert reason in evidence(audit)["rejection_reasons"]


def test_prompt_leakage_of_gold_text_is_rejected(tmp_path):
    case = build_case(tmp_path, LIMITS, GRID, [[], [0], [0], [0]], [AXIS])
    out, _ = run_case(tmp_path, [actuate("limit"), verdict("fail", [finding([0])])], LIMITS, case)
    prompt = out / "model_calls/000/prompt.txt"
    prompt.write_text(prompt.read_text() + "secret gold text")
    audit = audit_episode(out, case[0], case[1], manifest=case[2])
    assert "private_information_in_prompt:model_calls/000" in evidence(audit)["rejection_reasons"]


def test_fixed_protocol_is_label_independent_and_resets_between_joints():
    grid = [{"view": "front", "joint": None, "detent": None},
            {"view": "front", "joint": "a", "detent": "limit"},
            {"view": "front", "joint": "b", "detent": "zero"},
            {"view": "side", "joint": None, "detent": None}]
    assert fixed_protocol_actions(grid) == [
        {"type": "request_view", "view_id": "front"},
        {"type": "actuate_joint", "joint_id": "a", "detent": "limit"},
        {"type": "actuate_joint", "joint_id": "a", "detent": "rest"},
        {"type": "actuate_joint", "joint_id": "b", "detent": "zero"}]


def test_export_separates_action_and_diagnosis_supervision(tmp_path):
    case = build_case(tmp_path, LIMITS, GRID, [[], [0], [0], [0]], [AXIS])
    out, _ = run_case(tmp_path, [actuate("limit"), actuate("limit"), verdict("uncertain")], LIMITS, case)
    manifest = export_episodes([out], tmp_path / "export")
    counts = manifest["counts"]
    assert counts["action_rows"] == 1 and counts["action_rows_rejected"] == 1
    assert counts["terminal_rows"] == 0 and counts["diagnosis_rejected"] == 1
    rows = [json.loads(line) for line in (tmp_path / "export/train.actions.jsonl").read_text().splitlines()]
    assert rows[0]["episode_diagnosis_accepted"] is False
    assert rows[0]["action_new_objective_evidence"] == [0]


def test_action_qualification_does_not_depend_on_hidden_labels(tmp_path):
    for name, labels, exposures in (("clean", [], [[], [], [], []]), ("defect", [AXIS], [[], [0], [0], [0]])):
        root = tmp_path / name
        root.mkdir()
        case = build_case(root, LIMITS, GRID, exposures, labels)
        out, _ = run_case(root, [{"action": {"type": "request_view", "view_id": "front"}},
                                 actuate("limit"), verdict("uncertain")], LIMITS, case)
        assert export_episodes([out], root / "export")["counts"]["action_rows"] == 2


def test_export_blocks_test_split_and_post_audit_edits(tmp_path):
    case = build_case(tmp_path, LIMITS, GRID, [[], [0], [0], [0]], [AXIS])
    out, _ = run_case(tmp_path, [actuate("limit"), verdict("fail", [finding([0])])], LIMITS, case)
    episode = json.loads((out / "episode.json").read_text())
    episode["data_identity"]["split"] = "test"
    (out / "episode.json").write_text(json.dumps(episode))
    manifest = export_episodes([out], tmp_path / "export")
    rejected = json.loads((tmp_path / "export/rejected.jsonl").read_text())
    assert manifest["counts"]["episodes_blocked"] == 1
    assert {"episode_changed_after_audit", "test_or_nonformal_split_forbidden"} <= set(rejected["reasons"])
