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
    reviews = {row["case_id"]: row for row in read_jsonl(dataset / "human_review.jsonl")}
    records = []
    for case_id, review in sorted(reviews.items()):
        judgment = review["human_judgment"]
        if review["status"] != "human_review_complete" or judgment["verdict"] not in {"pass", "fail"}:
            raise ValueError(f"Incomplete binary human review: {case_id}")
        image = dataset / "inputs" / f"{case_id}.png"
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
