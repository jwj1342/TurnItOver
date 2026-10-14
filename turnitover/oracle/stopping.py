"""Visible-prefix stop/continue supervision with explicit policy semantics."""
from __future__ import annotations

import json
from pathlib import Path

from turnitover.oracle.episodes import TRAINING_OBJECTIVES, _termination, coverage
from turnitover.taxonomy import load_taxonomy


def _target(objective: str, evidence: dict, *, final_prefix: bool, remaining: int) -> tuple[dict, str]:
    has_evidence = bool(evidence["covered_label_indices"])
    policy_satisfied = has_evidence if objective == "first_failure" else final_prefix
    if policy_satisfied and has_evidence:
        return ({
            "stop_decision": {
                "decision": "stop",
                "outcome": "fail",
                "reason": (
                    "The current observations contain qualified defect evidence and satisfy "
                    "the selected stopping policy."
                ),
            }
        }, "evidence_sufficient")
    if not final_prefix and remaining > 0:
        return ({
            "stop_decision": {
                "decision": "continue",
                "outcome": None,
                "reason": (
                    "The current observations do not yet meet the selected stopping policy; "
                    "further inspection is required."
                ),
            }
        }, "further_evidence_required")
    return ({
        "stop_decision": {
            "decision": "stop",
            "outcome": "uncertain",
            "reason": (
                "No qualified defect has been established and no further observation is "
                "available under the selected policy."
            ),
        }
    }, "budget_exhausted_uncertain" if remaining == 0 else "observation_space_exhausted_uncertain")


def stopping_examples(uid, context, reference, history, witness_indices, exposures, labels, budget,
                      objective="first_failure"):
    """Return one stop/continue target for every causal prefix used by an Episode.

    Labels and exposure tables determine the audited target but never enter model messages.
    A budget-exhausted inconclusive prefix is retained as a separate uncertain stop outcome.
    """
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"Unknown Episode training objective: {objective}")
    if len(history) != len(witness_indices):
        raise ValueError("One witness index per observation is required")
    used, _ = _termination(labels, exposures, witness_indices, budget, objective)
    history = history[:used]
    witness_indices = witness_indices[:used]
    protocol = (Path(__file__).parents[1] / "verifier/prompt.txt").read_text(encoding="utf-8")
    instruction = (
        "\nThis example supervises only whether to continue observing or stop under the named "
        "policy. Return only a stop_decision JSON object.\n"
    )
    for decision_index in range(len(history) + 1):
        prefix = history[:decision_index]
        evidence = coverage(labels, exposures, witness_indices[:decision_index])
        remaining = max(0, budget - decision_index)
        final_prefix = decision_index == len(history)
        target, reason = _target(
            objective, evidence, final_prefix=final_prefix, remaining=remaining)
        images = [reference] + [row["image_ref"] for row in prefix]
        payload = {
            "context": context,
            "remaining": remaining,
            "history": prefix,
            "image_order": images,
            "taxonomy_ids": load_taxonomy().ids,
            "training_objective": objective,
        }
        yield {
            "id": f"{uid}:stop:{decision_index}",
            "episode_id": uid,
            "training_objective": objective,
            "decision_index": decision_index,
            "condition": "single_reference_stop_decision",
            "terminal": target["stop_decision"]["decision"] == "stop",
            "stop_target": target["stop_decision"]["decision"],
            "stop_outcome": target["stop_decision"]["outcome"],
            "stop_reason": reason,
            "evidence": evidence,
            "images": images,
            "messages": [
                {"role": "user", "content": protocol + instruction + json.dumps(payload)},
                {"role": "assistant", "content": json.dumps(target)},
            ],
        }
