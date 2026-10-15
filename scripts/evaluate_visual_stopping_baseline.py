#!/usr/bin/env python3
"""Fit the v5 no-image stage/budget majority baseline and score strict pairs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.oracle.stopping_metrics import summarize_pairs, summarize_stopping


FEATURES = ("decision_index", "remaining", "history_length", "image_count")


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _payload(record: dict) -> dict:
    return json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])


def _features(record: dict) -> tuple[int, int, int, int]:
    payload = _payload(record)
    return (record["decision_index"], payload["remaining"],
            len(payload["history"]), len(record["images"]))


def _majority(counts: Counter) -> tuple[str, str | None]:
    return sorted(counts, key=lambda label: (-counts[label], repr(label)))[0]


def evaluate(dataset: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    output.mkdir(parents=True)
    train, validation = _read(dataset / "train.stopping.jsonl"), _read(dataset / "validation.stopping.jsonl")
    pairs = _read(dataset / "validation.pairs.jsonl")
    lookup: dict[tuple, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for record in train:
        label = (record["stop_target"], record["stop_outcome"])
        lookup[_features(record)][label] += 1
        global_counts[label] += 1
    rows = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in validation:
            decision, outcome = _majority(lookup.get(_features(record)) or global_counts)
            row = {
                "id": record["id"], "episode_id": record["episode_id"],
                "asset_id": record["data_identity"]["asset_id"],
                "decision_index": record["decision_index"],
                "termination_kind": record.get("termination_kind"),
                "stop_reason": record["stop_reason"],
                "expected_decision": record["stop_target"],
                "expected_outcome": record["stop_outcome"],
                "predicted_decision": decision, "predicted_outcome": outcome,
                "features": dict(zip(FEATURES, _features(record), strict=True)),
                "image_bytes_read": False,
            }
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
    index = {record["id"]: record for record in validation}
    report = {
        "version": 1, "status": "complete", "scope": "v5_no_image_budget_baseline",
        "features": list(FEATURES), "image_bytes_read": False,
        "fit_split": "train", "evaluation_split": "validation",
        "dataset": str(dataset),
        "dataset_manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "overall": summarize_stopping(rows),
        "strict_pairs": summarize_pairs(rows, pairs, index),
        "frozen_test_used": False,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset.resolve(), args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
