"""Auditable pass eligibility from public single-reference Episode inputs only."""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image


FRONT_REST_SCOPE = (
    "Scope=front_rest_appearance_only. Compare only the sole front/rest reference "
    "with a candidate front/rest observation; do not judge articulation or unseen views."
)
RUNTIME_BUDGET_SCOPE = (
    "Scope=runtime_budget_only. Check only observed triangle and draw-call counts "
    "against the configured limits; do not judge geometry or articulation."
)


def _public_image(root: Path, relative: str) -> np.ndarray:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Public image references must stay within the Episode export")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Public image outside Episode export")
    with Image.open(io.BytesIO(resolved.read_bytes())) as image:
        return np.asarray(image.convert("RGB")).copy()


def assess_single_reference_pass(payload: dict, root: Path) -> dict:
    """Return scope-limited pass support without labels, gold code, or checker results."""
    context = payload["context"]
    task = context["task"]
    history = payload["history"]
    image_order = payload["image_order"]
    if not history:
        return {"eligible": False, "scope": None, "reason": "no_observations"}

    front = None
    articulated = False
    stats = None
    for observation in history:
        action = observation["action"]
        if action["type"] == "actuate_joint":
            articulated = True
        elif action["type"] == "request_view" and action.get("view_id") == "front" and not articulated and front is None:
            front = observation
        elif action["type"] == "query_runtime" and action.get("property") == "stats":
            stats = observation.get("payload")

    front_match = False
    if front is not None and front.get("image_ref") and image_order:
        reference = _public_image(root, image_order[0])
        candidate = _public_image(root, front["image_ref"])
        front_match = reference.shape == candidate.shape and np.array_equal(reference, candidate)

    runtime_complete = bool(stats) and all(
        type(stats.get(key)) in (int, float) and math.isfinite(stats[key]) and stats[key] >= 0
        for key in ("triangles_scene", "draw_calls")
    )
    facts = {"front_rest_exact_match": front_match, "runtime_stats_complete": runtime_complete}
    if task == FRONT_REST_SCOPE:
        return {**facts, "eligible": front_match, "scope": "front_rest_appearance_only",
                "reason": "complete_public_scope" if front_match else "front_rest_mismatch_or_missing"}
    if task == RUNTIME_BUDGET_SCOPE:
        within = (
            runtime_complete
            and stats["triangles_scene"] <= context["max_triangles"]
            and stats["draw_calls"] <= context["max_draw_calls"]
        )
        return {**facts, "eligible": within, "scope": "runtime_budget_only",
                "reason": "complete_public_scope" if within else "runtime_stats_missing_or_over_budget"}
    return {**facts, "eligible": False, "scope": None,
            "reason": "task_scope_not_limited_to_an_auditable_single_reference_rule"}
