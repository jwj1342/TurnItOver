"""Public-input records for the deliberately minimal Level 0 pass/fail task."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROMPT = (
    "The image is a four-panel comparison. Top row: normal reference at rest, then limit. "
    "Bottom row: candidate at rest, then limit. Does the candidate motion match the reference? "
    "Answer exactly one word: pass or fail."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_records(dataset: Path) -> list[dict]:
    """Load only opaque case IDs, pixels, and human-reviewed binary targets."""
    review_rows = read_jsonl(dataset / "human_review.jsonl")
    reviews = {row["case_id"]: row for row in review_rows}
    if len(reviews) != len(review_rows):
        raise ValueError("Duplicate human review case")
    manifest_path = dataset / "train_manifest.jsonl"
    manifest = read_jsonl(manifest_path) if manifest_path.exists() else None
    if manifest is not None:
        case_ids = [row["case_id"] for row in manifest]
        if len(set(case_ids)) != len(case_ids) or set(case_ids) - set(reviews):
            raise ValueError("Invalid training manifest membership")
        previews = {row["case_id"]: row["input_preview"] for row in manifest}
    else:
        case_ids = sorted(reviews)
        previews = {case: f"inputs/{case}.png" for case in case_ids}
    records = []
    for case_id in sorted(case_ids):
        review = reviews[case_id]
        judgment = review["human_judgment"]
        if review["status"] != "human_review_complete" or judgment["verdict"] not in {"pass", "fail"}:
            raise ValueError(f"Incomplete binary human review: {case_id}")
        image = dataset / previews[case_id]
        if not image.resolve().is_relative_to(dataset.resolve()):
            raise ValueError(f"Input path outside dataset: {case_id}")
        if not image.is_file():
            raise ValueError(f"Missing input image: {case_id}")
        records.append({"case_id": case_id, "image": image, "label": judgment["verdict"]})
    return records


def records_by_split(dataset: Path, split_path: Path, split: str) -> list[dict]:
    membership = json.loads(split_path.read_text())[split + "_case_ids"]
    records = {r["case_id"]: r for r in load_records(dataset)}
    if set(membership) - set(records):
        raise ValueError("Split references unknown case")
    return [records[case_id] for case_id in membership]
