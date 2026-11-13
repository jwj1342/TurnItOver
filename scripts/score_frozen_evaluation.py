#!/usr/bin/env python3
"""Recompute fixed-observation metrics from a run's private snapshot, without a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.oracle.evaluation_snapshot import load_snapshot
from turnitover.oracle.sft_metrics import public_input_digest
from turnitover.oracle.snapshot_scoring import score_snapshot
from turnitover.telemetry import git_state


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(evaluation, output):
    evaluation, output = Path(evaluation), Path(output)
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    source = json.loads((evaluation / "metrics.json").read_text())
    if source.get("status") != "complete":
        raise ValueError("Only complete evaluations can be scored")
    snapshot = load_snapshot(evaluation)
    manifest_sha = _sha(snapshot["root"] / "manifest.json")
    prediction_path = evaluation / "predictions.jsonl"
    prediction_sha = _sha(prediction_path)
    if (source.get("scoring_snapshot_sha256") != manifest_sha
            or source.get("predictions_sha256") != prediction_sha):
        raise ValueError("Evaluation snapshot or predictions hash mismatch")
    protocol = source["protocol"]
    manifest = snapshot["manifest"]
    if (manifest["task"] != protocol["task"] or manifest["split"] != protocol["split"]
            or manifest["defect_ids"] != protocol["defect_ids"]
            or manifest["case_count"] != protocol["n"]
            or manifest["source_sha256"]["dataset_manifest"] != protocol["dataset_manifest_sha256"]
            or manifest["source_sha256"]["annotations"] != protocol["annotations_sha256"]):
        raise ValueError("Snapshot differs from evaluation protocol")
    originals = {row["id"]: row["original_images"] for row in snapshot["image_paths"]}
    records = snapshot["records"]
    for record in records:
        record["images"] = originals[record["id"]]
    if public_input_digest(records) != protocol["public_input_sha256"]:
        raise ValueError("Snapshot public inputs differ from evaluation")
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
    report = score_snapshot(records, snapshot["annotations"], predictions,
                            manifest["defect_ids"], manifest["task"])
    report.update(status="complete", scoring_schema_version=1, protocol=protocol,
                  training_seed=source.get("training_seed"),
                  source_metrics_sha256=_sha(evaluation / "metrics.json"),
                  scoring_snapshot_sha256=manifest_sha, predictions_sha256=prediction_sha,
                  scorer_git=git_state(REPO_ROOT))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.evaluation.resolve(), args.out.resolve())
    print(json.dumps({"status": report["status"], "n": report["n"]}))


if __name__ == "__main__":
    main()
