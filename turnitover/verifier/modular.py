"""Protocol-preserving coordination for independent diagnosis and stopping modules."""
from __future__ import annotations

import copy
import json

from turnitover.verifier.contracts import Context, DecisionError, Verdict, parse_decision, parse_json


SUPPORTED_FINDINGS = {"structure.part_offset", "kinematics.joint_axis"}


def build_single_reference_record(template: dict, history: list[dict], remaining: int) -> dict:
    """Replace only the causal payload while retaining the frozen v4 prompt protocol."""
    record = copy.deepcopy(template)
    prefix, payload_text = record["messages"][0]["content"].rsplit("\n", 1)
    payload = json.loads(payload_text)
    public_history = []
    for index, observation in enumerate(history):
        public_history.append({
            "step": observation["step"],
            "action": observation["action"],
            "image_ref": f"observation_{index}.png",
            "payload": observation["payload"],
        })
    images = ["reference.png", *(row["image_ref"] for row in public_history)]
    payload.update(remaining=remaining, history=public_history, image_order=images)
    record["images"] = images
    record["messages"][0]["content"] = prefix + "\n" + json.dumps(payload)
    return record


def parse_stop(text: str) -> tuple[str, str | None]:
    value = parse_json(text)
    if set(value) != {"stop_decision"} or not isinstance(value["stop_decision"], dict):
        raise DecisionError("Expected one stop_decision object")
    decision = value["stop_decision"].get("decision")
    outcome = value["stop_decision"].get("outcome")
    if decision not in ("continue", "stop"):
        raise DecisionError("Invalid stop decision")
    if decision == "continue" and outcome is not None:
        raise DecisionError("Continue must have null outcome")
    if decision == "stop" and outcome not in ("fail", "uncertain"):
        raise DecisionError("Stop requires fail or uncertain outcome")
    return decision, outcome


def parse_verdict(text: str, context: dict, history: list[dict]) -> Verdict:
    return parse_decision(
        parse_json(text), Context(**context), {row["step"] for row in history}, final_only=True)


def finding_signature(finding) -> tuple[str, str | None, tuple[str, ...]]:
    return finding.defect_id, finding.joint_id, tuple(sorted(finding.parts))


def gate_findings(verdict: Verdict | None, history: list[dict], visual_evidence: list[dict],
                  *, require_aligned_visual: bool) -> tuple[list, list[dict]]:
    """Keep only claims supported by their cited, actually observed causal steps.

    The v5 gate establishes an aligned shape mismatch, not its defect type. The
    v4 diagnosis still supplies the structured finding. Axis claims additionally
    require a cited actuation of that same joint; a static mismatch cannot support
    a kinematic-axis conclusion.
    """
    if verdict is None or verdict.status != "fail":
        return [], []
    by_step = {row["step"]: row for row in history}
    accepted, rejected = [], []
    for finding in verdict.findings:
        reason = None
        if finding.defect_id not in SUPPORTED_FINDINGS:
            reason = "outside_validated_diagnosis_scope"
        cited = [step for step in finding.evidence_steps if step in by_step]
        if reason is None and not cited:
            reason = "no_observed_citation"
        if require_aligned_visual and reason is None:
            cited = [step for step in cited if visual_evidence[step]["state"] == "sufficient"]
            if not cited:
                reason = "no_cited_aligned_visual_mismatch"
        if finding.defect_id == "kinematics.joint_axis" and reason is None:
            cited = [step for step in cited
                     if by_step[step]["action"].get("type") == "actuate_joint"
                     and by_step[step]["action"].get("joint_id") == finding.joint_id]
            if not cited:
                reason = "axis_claim_lacks_cited_same_joint_actuation"
        if reason is None:
            accepted.append(finding)
        else:
            rejected.append({"signature": finding_signature(finding), "reason": reason,
                             "evidence_steps": list(finding.evidence_steps)})
    return accepted, rejected


def annotation_witness_indices(annotation: dict, history: list[dict],
                               reference_states: list[dict]) -> list[int | None]:
    """Map public v5 aligned states to the frozen v4 audit grid for scoring only."""
    grid = annotation["grid"]
    indices = []
    for observation in history:
        state = reference_states[observation["reference_index"]]
        target = ({"view": state["view"], "joint": None, "detent": None}
                  if state["detent"] == "rest" else
                  {"view": state["view"], "joint": state["joint"], "detent": state["detent"]})
        matches = [index for index, value in enumerate(grid) if value == target]
        if len(matches) > 1:
            raise ValueError(f"Aligned state has duplicate v4 witness matches: {target}")
        # v5 adds left/limit references absent from the frozen v4 evidence grid.
        # Preserve that protocol gap instead of assigning a private defect label.
        indices.append(matches[0] if matches else None)
    return indices


def visible_label_steps(annotation: dict, history: list[dict],
                        reference_states: list[dict]) -> dict[int, list[int]]:
    witnesses = annotation_witness_indices(annotation, history, reference_states)
    result = {index: [] for index in range(len(annotation["labels"]))}
    for observation, witness in zip(history, witnesses, strict=True):
        if witness is None:
            continue
        for label_index in annotation["exposures"][witness]:
            result[label_index].append(observation["step"])
    return result


def label_signature(label: dict) -> tuple[str, str | None, tuple[str, ...]]:
    joint = label.get("params", {}).get("joint") if label["defect_id"] == "kinematics.joint_axis" else None
    return label["defect_id"], joint, tuple(sorted(label["parts"]))
