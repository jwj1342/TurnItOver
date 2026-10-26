"""Release-boundary checks for the derived Easy candidate collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def validate_imported_review(gold: dict[str, Any], review: dict[str, Any]) -> None:
    """Allow genuine review progress without treating inclusion as qualification."""
    case_id = gold["case_id"]
    status = review.get("status")
    eligibility = gold.get("training_eligibility")
    if status not in {
        "pending_human_review", "human_review_complete", "human_review_rejected"
    }:
        raise ValueError(f"unknown imported review status: {case_id}")
    if eligibility not in {"pending_human_review", "eligible_individually_reviewed"}:
        raise ValueError(f"unsupported imported training eligibility: {case_id}")
    if status != "pending_human_review":
        if not review.get("reviewer") or not review.get("reviewed_at"):
            raise ValueError(f"completed imported review lacks attribution: {case_id}")
    if eligibility == "eligible_individually_reviewed":
        judgment = review.get("human_judgment", {})
        if (
            status != "human_review_complete"
            or review.get("review_mode") not in {
                "user_per_case_blind_review", "independent_per_case_blind_review"
            }
            or not isinstance(judgment, dict)
            or gold.get("expected_status") not in {"pass", "fail"}
            or judgment.get("verdict") != gold["expected_status"]
        ):
            raise ValueError(f"imported case is not review-gated: {case_id}")


def validate_blind_review_sources(root: Path, reviews: dict[str, dict[str, Any]]) -> None:
    """Bind distributable media provenance to the verified source collection."""
    bundle = root / "validation_review"
    manifest = json.loads((bundle / "case_manifest.json").read_text(encoding="utf-8"))
    expected = {}
    for row in manifest:
        source = reviews[row["case_id"]]
        paths = [("board", source["board"], row["board"]),
                 ("input_preview", source["input_preview"], row["input_preview"])]
        paths += [(f"image_{i}", before, after)
                  for i, (before, after) in enumerate(zip(source["images"], row["images"]))]
        for role, before, after in paths:
            expected[after] = (row["case_id"], role, before)
    provenance = json.loads((bundle / "media_provenance.json").read_text(encoding="utf-8"))
    for row in provenance:
        identity = (row["case_id"], row["role"], row["source_path"])
        if expected[row["derived_path"]] != identity:
            raise ValueError("blind media source identity differs from root review manifest")
        source_hash = hashlib.sha256((root / row["source_path"]).read_bytes()).hexdigest()
        if row["source_sha256"] != source_hash:
            raise ValueError("blind media source hash differs from root media")
        if row["role"] != "board" and row["derived_sha256"] != source_hash:
            raise ValueError("uncropped blind media differs from root media")
