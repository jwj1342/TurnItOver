"""Recompute the fixed Level 0 comparison from complete per-case artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from turnitover.oracle.level0_axis import PROMPT, read_jsonl, records_by_split, sha256


def summarize(dataset: Path, split: Path, root: Path) -> dict:
    digest = sha256(split)
    objects = json.loads((dataset / "object_splits.json").read_text())["objects"]
    assets = {case: obj["asset_id"] for obj in objects for case in obj["case_ids"]}
    result = {"split_sha256": digest, "methods": {}, "training": {}}
    for method in ("base", "rank4", "rank8", "pixel"):
        cohorts = {}
        for cohort in ("train", "heldout"):
            directory = root / method
            if method.startswith("rank"):
                directory /= "eval"
            if method != "pixel":
                directory /= cohort
            rows = read_jsonl(directory / "predictions.jsonl")
            if method == "pixel":
                rows = [row for row in rows if row["cohort"] == cohort]
            expected = {r["case_id"]: r["label"] for r in records_by_split(dataset, split, cohort)}
            if len(rows) != len(expected) or {r["case_id"] for r in rows} != set(expected):
                raise ValueError(f"Incomplete/duplicate predictions: {method}/{cohort}")
            for row in rows:
                assert row["expected"] == expected[row["case_id"]]
                assert row["correct"] == (row["predicted"] == row["expected"])
            metrics = json.loads((directory / "metrics.json").read_text())
            assert metrics["status"] == "complete"
            assert metrics["split_sha256" if method == "pixel" else "dataset_split_sha256"] == digest
            score = {}
            for part, label in (("overall", None), ("clean", "pass"), ("axis", "fail")):
                selected = [r for r in rows if label is None or r["expected"] == label]
                correct = sum(r["correct"] for r in selected)
                score[part] = {"correct": correct, "n": len(selected), "accuracy": correct / len(selected)}
            recorded = metrics["cohorts"][cohort] if method == "pixel" else metrics
            assert score["overall"]["correct"] == recorded["correct"]
            score["errors"] = [dict(r, asset_id=assets[r["case_id"]]) for r in rows if not r["correct"]]
            cohorts[cohort] = score
        result["methods"][method] = cohorts
    for rank in (4, 8):
        directory = root / f"rank{rank}" / "train"
        run = json.loads((directory / "run.json").read_text())
        losses = read_jsonl(directory / "loss.jsonl")
        assert run["status"] == "complete" and run["steps"] == len(losses) == 132
        assert run["split_sha256"] == digest and run["initial_adapter"] is None
        assert run["prompt"] == PROMPT and run["seed"] == 20261017
        assert run["lora"]["rank"] == rank and run["lora"]["alpha"] == 2 * rank
        assert run["epochs"] == 3 and run["learning_rate"] == 2e-4 and run["batch_size"] == 1
        assert [row["step"] for row in losses] == list(range(132))
        for epoch in range(3):
            rows = [r for r in losses if r["epoch"] == epoch]
            assert len(rows) == 44 and {r["case_id"] for r in rows} == set(
                json.loads(split.read_text())["train_case_ids"])
        result["training"][f"rank{rank}"] = {
            key: run[key] for key in ("seconds", "peak_allocated_bytes", "trainable_parameters", "checkpoint_sha256")}
        result["training"][f"rank{rank}"]["epoch_mean_loss"] = [
            sum(r["loss"] for r in losses if r["epoch"] == epoch) / 44 for epoch in range(3)]
    # Pair ranks on exactly the same example order, preprocessing and optimizer.
    a, b = [json.loads((root / f"rank{rank}/train/run.json").read_text()) for rank in (4, 8)]
    for key in ("seed", "epochs", "steps", "learning_rate", "optimizer", "scheduler",
                "max_grad_norm", "batch_size", "base_model", "base_model_config_sha256", "visual_preprocess"):
        assert a[key] == b[key], key
    orders = [[(r["epoch"], r["case_id"]) for r in read_jsonl(root / f"rank{rank}/train/loss.jsonl")]
              for rank in (4, 8)]
    assert orders[0] == orders[1]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    (args.root / "summary.json").write_text(json.dumps(summarize(args.dataset, args.split, args.root), indent=2) + "\n")
