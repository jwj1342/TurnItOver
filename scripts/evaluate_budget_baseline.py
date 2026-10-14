#!/usr/bin/env python3
"""Fit and evaluate a no-image metadata baseline on an existing Episode split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


FEATURE_NAMES = ("step", "remaining", "history_length", "image_count")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _payload(record: dict) -> dict:
    return json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])


def features(record: dict) -> tuple[int, int, int, int]:
    payload = _payload(record)
    history_length = len(payload["history"])
    return record["decision_index"], payload["remaining"], history_length, len(record["images"])


class MajorityLookup:
    """Deterministic empirical lookup with a global-majority fallback."""

    def __init__(self):
        self.by_key: dict[tuple, Counter] = defaultdict(Counter)
        self.global_counts = Counter()

    def fit_one(self, key: tuple, label) -> None:
        self.by_key[key][label] += 1
        self.global_counts[label] += 1

    @staticmethod
    def _majority(counts: Counter):
        if not counts:
            raise ValueError("Cannot predict without fitted labels")
        return sorted(counts, key=lambda label: (-counts[label], repr(label)))[0]

    def predict(self, key: tuple):
        return self._majority(self.by_key.get(key) or self.global_counts)


def _annotation_index(dataset: Path, split: str) -> dict[str, dict]:
    return {row["id"]: row for row in _rows(dataset / "annotations.private.jsonl")
            if row["split"] == split}


def _expected(record: dict, annotation: dict) -> tuple[
        str, tuple[str, ...], tuple[tuple[str, ...], ...], tuple[tuple, ...]]:
    covered = record["evidence"]["covered_label_indices"]
    labels = [annotation["labels"][index] for index in covered]
    status = "fail" if labels else "uncertain"
    defect_types = tuple(sorted({row["defect_id"] for row in labels}))
    parts = tuple(sorted({tuple(sorted(row["parts"])) for row in labels}))
    signatures = tuple(sorted((
        row["defect_id"],
        row.get("params", {}).get("joint") if row["defect_id"] == "kinematics.joint_axis" else None,
        tuple(sorted(row["parts"])),
    ) for row in labels))
    return status, defect_types, parts, signatures


def _stop_index(dataset: Path, split: str) -> dict[tuple[str, int], dict]:
    rows = _rows(dataset / f"{split}.stopping.jsonl")
    index = {(row["episode_id"], row["decision_index"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate stop-supervision prefix")
    return index


def evaluate(dataset: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("formal_training") is not True:
        raise ValueError("Complete formal-training supervision is required")
    output.mkdir(parents=True)

    prefix_status_model, terminal_status_model = MajorityLookup(), MajorityLookup()
    prefix_defect_model, terminal_defect_model = MajorityLookup(), MajorityLookup()
    stop_model = MajorityLookup()
    train_annotations = _annotation_index(dataset, "train")
    train_stops = _stop_index(dataset, "train")
    for record in _rows(dataset / "train.episodes.jsonl"):
        key = features(record)
        status, defects, _, _ = _expected(record, train_annotations[record["episode_id"]])
        prefix_status_model.fit_one(key, status)
        prefix_defect_model.fit_one(key, defects)
        if record["terminal"]:
            terminal_status_model.fit_one(key, status)
            terminal_defect_model.fit_one(key, defects)
        stop = train_stops[(record["episode_id"], record["decision_index"])]["stop_target"]
        stop_model.fit_one(key, stop)

    validation_annotations = _annotation_index(dataset, "validation")
    validation_stops = _stop_index(dataset, "validation")
    validation = _rows(dataset / "validation.episodes.jsonl")
    counts = Counter(n=len(validation))
    per_asset: dict[str, Counter] = defaultdict(Counter)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in validation:
            key = features(record)
            expected_status, expected_defects, expected_parts, expected_findings = _expected(
                record, validation_annotations[record["episode_id"]])
            expected_stop = validation_stops[
                (record["episode_id"], record["decision_index"])]["stop_target"]
            status_model = terminal_status_model if record["terminal"] else prefix_status_model
            defect_model = terminal_defect_model if record["terminal"] else prefix_defect_model
            predicted_status = status_model.predict(key)
            predicted_defects = defect_model.predict(key)
            predicted_stop = stop_model.predict(key)
            asset_id = record["data_identity"]["asset_id"]
            terminal = record["terminal"]
            status_correct = predicted_status == expected_status
            stop_correct = predicted_stop == expected_stop
            counts["status_correct"] += status_correct
            counts["stop_correct"] += stop_correct
            counts["terminal_status_correct"] += terminal and status_correct
            counts["terminal_n"] += terminal
            counts["defect_type_exact"] += predicted_defects == expected_defects
            counts["terminal_defect_type_exact"] += terminal and predicted_defects == expected_defects
            if terminal:
                expected_type_set, predicted_type_set = set(expected_defects), set(predicted_defects)
                counts["terminal_type_tp"] += len(expected_type_set & predicted_type_set)
                counts["terminal_type_fp"] += len(predicted_type_set - expected_type_set)
                counts["terminal_type_fn"] += len(expected_type_set - predicted_type_set)
            counts["no_evidence_correct"] += not expected_defects and predicted_status == "uncertain"
            counts["no_evidence_n"] += not expected_defects
            counts["false_fail"] += not expected_defects and predicted_status == "fail"
            bucket = per_asset[asset_id]
            bucket["n"] += 1
            bucket["status_correct"] += status_correct
            bucket["stop_correct"] += stop_correct
            row = {
                "id": record["id"], "episode_id": record["episode_id"],
                "asset_id": asset_id, "condition": "no_image_budget_baseline",
                "terminal": terminal, "decision_index": record["decision_index"],
                **dict(zip(FEATURE_NAMES, key, strict=True)),
                "evidence_status": record["evidence"]["status"],
                "evidence_available": bool(expected_defects),
                "expected_stop_decision": expected_stop,
                "predicted_stop_decision": predicted_stop,
                "decision_kind": "verdict" if terminal or predicted_stop == "stop" else "action",
                "expected_status": expected_status,
                "predicted_status": predicted_status,
                "expected_defect_types": list(expected_defects),
                "predicted_defect_types": list(predicted_defects),
                "source_defects": list(expected_defects),
                "defect_combination": "+".join(expected_defects) if expected_defects else "no_visible_evidence",
                "expected_parts": [list(parts) for parts in expected_parts],
                "predicted_parts": [],
                "expected_findings": [[defect, joint, list(parts)]
                                      for defect, joint, parts in expected_findings],
                "predicted_findings": [],
                "unsupported_findings": len(set(predicted_defects) - set(expected_defects)),
                "image_bytes_read": False,
            }
            stream.write(json.dumps(row) + "\n")

    report = {
        "version": 1, "status": "complete", "scope": "no_image_metadata_baseline",
        "features": list(FEATURE_NAMES), "image_bytes_read": False,
        "fit_protocol": "Terminal diagnosis heads fit terminal Train rows; stop head fits every Train prefix.",
        "fit_split": "train", "evaluation_split": "validation",
        "n": counts["n"], "terminal_n": counts["terminal_n"],
        "status_accuracy_all_prefixes": counts["status_correct"] / counts["n"],
        "status_accuracy_terminal": counts["terminal_status_correct"] / counts["terminal_n"],
        "stop_decision_accuracy": counts["stop_correct"] / counts["n"],
        "defect_type_exact_accuracy": counts["defect_type_exact"] / counts["n"],
        "terminal_defect_type_exact_accuracy": counts["terminal_defect_type_exact"] / counts["terminal_n"],
        "terminal_defect_type_counts": {
            "tp": counts["terminal_type_tp"], "fp": counts["terminal_type_fp"],
            "fn": counts["terminal_type_fn"],
        },
        "no_evidence_uncertain_accuracy": (
            counts["no_evidence_correct"] / counts["no_evidence_n"] if counts["no_evidence_n"] else None),
        "false_fail_count": counts["false_fail"],
        "part_localization": {"tp": 0, "note": "Part identity is excluded from baseline inputs."},
        "per_asset": {key: dict(value) for key, value in sorted(per_asset.items())},
        "dataset": str(dataset),
        "dataset_manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "private_training_labels_excluded_from_features": True,
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
