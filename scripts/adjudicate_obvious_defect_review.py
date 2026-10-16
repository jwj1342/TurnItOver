#!/usr/bin/env python3
"""Validate blind human reviews and compare completed judgments with private Gold."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "docs/evidence/partnet-mobility-obvious-defects-v1"
DEFAULT_REVIEW = ROOT / "docs/evidence/partnet-mobility-obvious-defects-v1-human-review"
STATUSES = {"human_review_complete", "needs_adjudication"}
VERDICTS = {"pass", "fail", "uncertain"}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(row: dict) -> dict:
    judgment = row["human_judgment"]
    return {"status": row["status"], **{key: judgment[key]
            for key in ("verdict", "defect_id", "joint_id", "parts", "evidence_states")}}


def validate_review(row: dict, public: dict) -> None:
    if row["status"] not in STATUSES or row["human_judgment"]["verdict"] not in VERDICTS:
        raise ValueError(f"Invalid review status or verdict: {row['case_id']}")
    judgment = row["human_judgment"]
    states = judgment["evidence_states"]
    if len(states) != len(set(states)) or not set(states) <= set(public["states"]):
        raise ValueError(f"Invalid evidence states: {row['case_id']}")
    if row["status"] == "human_review_complete" and judgment["verdict"] not in {"pass", "fail"}:
        raise ValueError(f"Completed review needs pass/fail: {row['case_id']}")
    if row["status"] == "needs_adjudication" and judgment["verdict"] != "uncertain":
        raise ValueError(f"Pending adjudication needs uncertain: {row['case_id']}")
    if judgment["verdict"] in {"pass", "uncertain"} and any(
            (judgment["defect_id"] is not None, judgment["joint_id"] is not None, judgment["parts"])):
        raise ValueError(f"Pass/uncertain review must not assert a defect: {row['case_id']}")
    if judgment["verdict"] == "fail" and (not judgment["defect_id"] or not judgment["parts"] or not states):
        raise ValueError(f"Fail review needs type, part and evidence: {row['case_id']}")


def adjudicate(bundle: Path, review_root: Path) -> tuple[list[dict], dict]:
    public_rows = rows(bundle / "review_manifest.jsonl")
    review_rows = rows(review_root / "human_review.jsonl")
    raw_rows = rows(review_root / "submissions.raw.jsonl")
    gold_rows = rows(bundle / "private/gold.jsonl")
    check_rows = rows(bundle / "private/checks.jsonl")
    public = {row["case_id"]: row for row in public_rows}
    reviews = {row["case_id"]: row for row in review_rows}
    gold = {row["case_id"]: row for row in gold_rows}
    checks = {row["case_id"]: row for row in check_rows}
    ids = set(public)
    if any(len(table) != len(ids) or set(table) != ids for table in (reviews, gold, checks)):
        raise ValueError("Public, review, Gold and checker case IDs must match exactly")

    raw_by_id = defaultdict(list)
    for row in raw_rows:
        if row["case_id"] not in ids:
            raise ValueError(f"Unknown raw review case: {row['case_id']}")
        raw_by_id[row["case_id"]].append(row)
    if set(raw_by_id) != ids:
        raise ValueError("Raw submissions must cover every case")
    duplicate_ids, conflicting_ids = {}, []
    for case_id, submitted in raw_by_id.items():
        signatures = {json.dumps(signature(row), sort_keys=True) for row in submitted}
        if len(submitted) > 1:
            duplicate_ids[case_id] = len(submitted)
        if len(signatures) != 1 or signature(reviews[case_id]) != signature(submitted[0]):
            conflicting_ids.append(case_id)
    if conflicting_ids:
        raise ValueError(f"Conflicting or mis-normalized submissions: {sorted(conflicting_ids)}")

    results = []
    for case_id in sorted(ids):
        review = reviews[case_id]
        validate_review(review, public[case_id])
        judgment, expected = review["human_judgment"], gold[case_id]
        checker_expected = expected["kind"] == "clean"
        checker_consistent = checks[case_id]["code_correct"] == checker_expected
        field_matches = {
            "verdict": judgment["verdict"] == expected["expected_status"],
            "defect_id": judgment["defect_id"] == expected["defect_id"],
            "joint_id": judgment["joint_id"] == expected["joint_id"],
            "parts": sorted(judgment["parts"]) == sorted(expected["parts"]),
        }
        complete = review["status"] == "human_review_complete"
        approved = complete and all(field_matches.values()) and checker_consistent
        reasons = []
        if not complete:
            reasons.append("human_uncertain")
        if complete:
            reasons.extend(f"human_gold_mismatch:{key}" for key, matched in field_matches.items() if not matched)
        if not checker_consistent:
            reasons.append("checker_gold_mismatch")
        results.append({
            "case_id": case_id, "human_status": review["status"],
            "human_judgment": judgment, "private_gold": {
                "expected_status": expected["expected_status"], "defect_id": expected["defect_id"],
                "joint_id": expected["joint_id"], "parts": expected["parts"]},
            "field_matches": field_matches, "checker_consistent": checker_consistent,
            "review_gate": "passed" if approved else "held",
            "candidate_status": "review_approved_candidate" if approved else "needs_adjudication",
            "reasons": reasons,
        })

    counts = Counter(row["candidate_status"] for row in results)
    by_kind = defaultdict(Counter)
    for row in results:
        by_kind[gold[row["case_id"]]["kind"]][row["candidate_status"]] += 1
    summary = {
        "version": 1, "status": "complete", "scope": "human_review_gate_only",
        "cases": len(results), "submission_rows": len(raw_rows),
        "duplicate_ids": duplicate_ids, "conflicting_duplicate_ids": [],
        "counts": dict(sorted(counts.items())),
        "by_gold_kind": {kind: dict(sorted(values.items())) for kind, values in sorted(by_kind.items())},
        "boundaries": {"sft_exported": False, "sft_trained": False, "test_modified": False,
                       "model_calls": 0},
        "inputs": {"parent_manifest_sha256": sha256(bundle / "manifest.json"),
                   "human_review_sha256": sha256(review_root / "human_review.jsonl"),
                   "raw_submissions_sha256": sha256(review_root / "submissions.raw.jsonl"),
                   "private_gold_sha256": sha256(bundle / "private/gold.jsonl"),
                   "private_checks_sha256": sha256(bundle / "private/checks.jsonl")},
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    results, summary = adjudicate(args.bundle.resolve(), args.review.resolve())
    private = args.review.resolve() / "private"
    private.mkdir(parents=True, exist_ok=True)
    (private / "adjudication.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results), encoding="utf-8")
    write_json_atomic(args.review.resolve() / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "cases": summary["cases"],
                      "counts": summary["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
