"""Unified action and terminal-verdict supervision from an existing oracle trace."""
from __future__ import annotations

import json
from pathlib import Path

from turnitover.oracle.supervision import action_examples
from turnitover.taxonomy import load_taxonomy
from turnitover.verifier.contracts import Context, parse_decision


def _coverage(labels, exposures, witness_indices):
    covered = set()
    for index in witness_indices:
        if type(index) is not int or not 0 <= index < len(exposures):
            raise ValueError("Trajectory references an unknown witness")
        covered.update(exposures[index])
    expected = set(range(len(labels)))
    if not covered <= expected:
        raise ValueError("Exposure references an unknown label")
    status = "complete" if covered == expected else "partial" if covered else "none"
    return {"status": status, "covered_label_indices": sorted(covered),
            "uncovered_label_indices": sorted(expected - covered),
            "proxy": "counterfactual_pixel_exposure"}


def _finding(label, steps):
    defect_id = label["defect_id"]
    if defect_id == "kinematics.joint_axis":
        description = "The part moves along a different axis from the reference articulation."
        fix = "Correct the joint axis to match the reference motion."
        joint_id = label.get("params", {}).get("joint")
    elif defect_id == "structure.part_offset":
        description = "The part is displaced relative to the reference configuration."
        fix = "Correct the part placement relative to its parent."
        joint_id = None
    else:
        raise ValueError(f"Unsupported evidence-qualified diagnosis: {defect_id}")
    return {"defect_id": defect_id, "parts": list(label["parts"]), "joint_id": joint_id,
            "severity": label["severity"], "confidence": 1.0, "evidence_steps": steps,
            "description": description, "suggested_fix": fix}


def _terminal_verdict(labels, exposures, history, witness_indices):
    evidence = _coverage(labels, exposures, witness_indices)
    if not history:
        raise ValueError("A terminal verdict requires at least one observation")
    findings = []
    for label_index in evidence["covered_label_indices"]:
        steps = [observation["step"] for observation, witness in zip(
            history, witness_indices, strict=True) if label_index in exposures[witness]]
        findings.append(_finding(labels[label_index], steps))
    if findings:
        limitations = ["The verdict covers only the acquired views and articulated states."]
        if evidence["status"] == "partial":
            limitations.append("Additional injected defects were not exposed by this trajectory.")
        verdict = {"status": "fail", "confidence": 1.0, "findings": findings,
                   "summary": "The acquired observations expose one or more qualified defects.",
                   "limitations": limitations}
    elif labels:
        verdict = {"status": "uncertain", "confidence": 0.0, "findings": [],
                   "summary": "The trajectory does not expose the injected defects.",
                   "limitations": ["No injected defect reached the configured pixel-exposure threshold."]}
    else:
        verdict = {"status": "pass", "confidence": 1.0, "findings": [],
                   "summary": "No defect exists within the configured synthetic constraints.",
                   "limitations": ["Pass is limited to the acquired observations and configured constraints."]}
    return verdict, evidence


def episode_examples(uid, context, reference, history, witness_indices, exposures, labels, budget):
    """Return tokenizer-compatible decisions grouped by one privileged teacher episode.

    Private labels and exposure tables remain outside the model messages. The final
    verdict is the explicit termination target; no separate stop action is introduced.
    """
    if len(history) != len(witness_indices):
        raise ValueError("One witness index per observation is required")
    rows = []
    for decision_index, row in enumerate(action_examples(uid, context, reference, history, budget)):
        rows.append({**row, "episode_id": uid, "decision_index": decision_index,
                     "terminal": False,
                     "evidence": _coverage(labels, exposures, witness_indices[:decision_index])})

    verdict, evidence = _terminal_verdict(labels, exposures, history, witness_indices)
    target = {"verdict": verdict}
    parse_decision(target, Context(**context), {row["step"] for row in history}, final_only=True)
    images = [reference] + [row["image_ref"] for row in history]
    payload = {"context": context, "remaining": max(0, budget - len(history)),
               "history": history, "image_order": images, "taxonomy_ids": load_taxonomy().ids}
    protocol = (Path(__file__).parents[1] / "verifier/prompt.txt").read_text(encoding="utf-8")
    rows.append({
        "id": f"{uid}:verdict", "episode_id": uid, "decision_index": len(history),
        "condition": "single_reference_episode_terminal", "terminal": True,
        "termination_target": "verdict", "evidence": evidence, "images": images,
        "messages": [
            {"role": "user", "content": protocol + "\n" + json.dumps(payload)},
            {"role": "assistant", "content": json.dumps(target)},
        ],
    })
    return rows
