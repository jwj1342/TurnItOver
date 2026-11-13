"""Three fixed reference/candidate observations shared by SFT and inference."""
from __future__ import annotations

import json
from pathlib import Path

from turnitover.oracle.episodes import _terminal_verdict
from turnitover.taxonomy import load_taxonomy
from turnitover.verifier.contracts import Context, parse_decision


OBSERVATION_FORMAT = "reference_candidate_pairs_v1"
STATES = ("zero", "mid", "limit")


def validate_pairs(payload, images):
    """Validate the ordered six-image, one-view public observation contract."""
    if payload.get("observation_format") != OBSERVATION_FORMAT:
        raise ValueError("Unknown paired observation format")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != len(STATES):
        raise ValueError("Paired observations require three state pairs")
    flattened, views = [], []
    for index, (row, state) in enumerate(zip(history, STATES, strict=True)):
        if (set(row) != {"step", "state", "action", "reference_image_ref", "image_ref"}
                or type(row["step"]) is not int or row["step"] != index or row["state"] != state):
            raise ValueError("Paired observations must be steps 0/1/2 at zero/mid/limit")
        action = row["action"]
        if (not isinstance(action, dict) or set(action) != {"type", "view_id"}
                or action["type"] != "request_view" or not isinstance(action["view_id"], str)
                or not action["view_id"]):
            raise ValueError("Paired observations need an explicit view")
        views.append(action["view_id"])
        flattened.extend((row["reference_image_ref"], row["image_ref"]))
    if any(not isinstance(path, str) or not path for path in flattened):
        raise ValueError("Paired observations need reference and candidate image paths")
    if len(set(flattened)) != 6 or flattened != images:
        raise ValueError("Paired image order must be reference/candidate for each state")
    if len(set(views)) != 1 or payload["context"].get("views") != [views[0]]:
        raise ValueError("Paired observations require one shared view")
    if payload.get("image_order") != images:
        raise ValueError("Public image order differs from paired image roster")


def paired_terminal(uid, context, history, exposures, labels):
    """Build a terminal diagnosis whose evidence indices identify image pairs."""
    if len(exposures) != len(STATES):
        raise ValueError("One exposure table per state pair is required")
    images = [path for row in history for path in (row["reference_image_ref"], row["image_ref"])]
    payload = {"context": context, "remaining": 0, "history": history,
               "image_order": images, "taxonomy_ids": load_taxonomy().ids,
               "training_objective": "comprehensive_within_budget",
               "observation_format": OBSERVATION_FORMAT}
    validate_pairs(payload, images)
    verdict, evidence = _terminal_verdict(labels, exposures, history, list(range(len(history))))
    evidence["pair_label_indices"] = [list(indices) for indices in exposures]
    pair_support(evidence, len(labels), evidence["covered_label_indices"])
    target = {"verdict": verdict}
    parse_decision(target, Context(**context), set(range(len(history))), final_only=True)
    protocol = (Path(__file__).parents[1] / "verifier/prompt.txt").read_text(encoding="utf-8")
    protocol = protocol.replace(
        "Reference images come first; acquired observation images follow in the explicitly listed order.",
        "Each observation is an independent [reference, candidate] image pair at the same view and state. "
        "The three pairs are zero, mid and limit, with reference first within each pair. "
        "There is no separate shared reference image. evidence_steps indexes the complete pair "
        "(0=zero, 1=mid, 2=limit), not an individual image.")
    return {"id": f"{uid}:verdict", "episode_id": uid, "terminal": True,
            "training_objective": "comprehensive_within_budget", "decision_index": 3,
            "condition": "paired_reference_candidate_terminal",
            "termination_target": "verdict", "termination_reason": "budget_exhausted",
            "evidence": evidence, "images": images,
            "messages": [{"role": "user", "content": protocol + "\n" + json.dumps(payload)},
                         {"role": "assistant", "content": json.dumps(target)}]}



def pair_support(evidence, label_count, covered):
    """Audit private per-pair exposures and return supporting steps per label."""
    pairs = evidence.get("pair_label_indices")
    if not isinstance(pairs, list) or len(pairs) != len(STATES):
        raise ValueError("Paired evidence needs three private exposure lists")
    support = {}
    for step, indices in enumerate(pairs):
        if (not isinstance(indices, list)
                or any(type(index) is not int or not 0 <= index < label_count for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("Invalid paired exposure label indices")
        for index in indices:
            support.setdefault(index, set()).add(step)
    if set(support) != set(covered):
        raise ValueError("Paired exposure coverage differs from covered labels")
    return support


def is_paired(record):
    try:
        payload = json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])
    except (ValueError, KeyError, IndexError, TypeError):
        return False
    return isinstance(payload, dict) and "observation_format" in payload


def user_content(record, images):
    """Return the same role-labelled image/text sequence for local and API models."""
    if len(images) != len(record["images"]):
        raise ValueError("Image roster mismatch")
    if not is_paired(record):
        return ([{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": record["messages"][0]["content"]}])
    from turnitover.oracle.supervision_validation import public_payload
    payload = public_payload(record)
    content = [{"type": "text", "text": record["messages"][0]["content"]}]
    for index, row in enumerate(payload["history"]):
        content.extend([
            {"type": "text", "text": f"Observation {index}, state={row['state']}, view={row['action']['view_id']}: reference"},
            {"type": "image", "image": images[2 * index]},
            {"type": "text", "text": f"Observation {index}: candidate"},
            {"type": "image", "image": images[2 * index + 1]},
        ])
    return content
