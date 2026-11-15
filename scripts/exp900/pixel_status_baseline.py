#!/usr/bin/env python3
"""Status-only pixel-difference baseline for the three-tier-900 fixed-observation export.

For every episode, pair each reference image with the candidate image of the same view and
articulation state, take the mean absolute RGB difference per pair, and use the maximum over
pairs as the episode score. A single threshold is fitted on the train split to maximise
status accuracy against the target verdict status ("fail" when the fixed trajectory exposes a
defect, "uncertain" otherwise, which includes clean candidates and defects the trajectory did
not expose) and is then applied unchanged to the validation split. The baseline names no
defect, part or evidence step, so it is scored on verdict status only and is not comparable on F1.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

PAIR = re.compile(r"^(?P<idx>\d+)-(?P<rest>.*)$")


def episode_score(root: Path, images: list[str]) -> tuple[float, int]:
    by_key = {}
    for rel in images:
        name = Path(rel).stem
        m = PAIR.match(name)
        if not m:
            raise ValueError(f"Unexpected image name {rel}")
        by_key[m.group("rest")] = rel
    diffs = []
    for key, rel in by_key.items():
        if "reference" not in key:
            continue
        partner = key.replace("reference", "candidate")
        if partner not in by_key:
            raise ValueError(f"No candidate image for {rel}")
        a = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.float32) / 255.0
        b = np.asarray(Image.open(root / by_key[partner]).convert("RGB"), dtype=np.float32) / 255.0
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch for {rel}")
        diffs.append(float(np.abs(a - b).mean()))
    if not diffs:
        raise ValueError("Episode has no reference/candidate pairs")
    return max(diffs), len(diffs)


def load(root: Path, split: str) -> list[dict]:
    rows = []
    for line in (root / f"{split}.episodes.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        target = json.loads(record["messages"][1]["content"])["verdict"]["status"]
        if target not in ("fail", "uncertain"):
            raise ValueError(f"Unexpected target status {target} in {record['id']}")
        score, pairs = episode_score(root, record["images"])
        rows.append({"id": record["id"], "episode_id": record["episode_id"], "difficulty": record["difficulty"],
                     "object": record["data_identity"]["source_identity_sha256"],
                     "expected_status": target, "score": score, "pairs": pairs})
    return rows


def fit_threshold(rows: list[dict]) -> float:
    scores = sorted({row["score"] for row in rows})
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 1e-6]
    best = max(candidates, key=lambda t: (accuracy(rows, t), -t))
    return best


def predict(row: dict, threshold: float) -> str:
    return "fail" if row["score"] > threshold else "uncertain"


def accuracy(rows: list[dict], threshold: float) -> float:
    return sum(predict(r, threshold) == r["expected_status"] for r in rows) / len(rows)


def summarize(rows: list[dict], threshold: float) -> dict:
    n = len(rows)
    fail_rows = [r for r in rows if r["expected_status"] == "fail"]
    uncertain_rows = [r for r in rows if r["expected_status"] == "uncertain"]
    return {"n": n, "independent_objects": len({r["object"] for r in rows}),
            "status_accuracy": accuracy(rows, threshold),
            "fail_cases": len(fail_rows),
            "fail_predicted_uncertain": sum(predict(r, threshold) == "uncertain" for r in fail_rows),
            "uncertain_cases": len(uncertain_rows),
            "uncertain_predicted_fail": sum(predict(r, threshold) == "fail" for r in uncertain_rows),
            "always_fail_status_accuracy": len(fail_rows) / n,
            "always_uncertain_status_accuracy": len(uncertain_rows) / n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Output must not exist: {args.out}")
    train = load(args.dataset, "train")
    validation = load(args.dataset, "validation")
    threshold = fit_threshold(train)
    args.out.mkdir(parents=True)
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in validation:
            stream.write(json.dumps({**row, "predicted_status": predict(row, threshold)}) + "\n")
    report = {"scope": "status_only_pixel_difference", "threshold": threshold,
              "score": "max over reference/candidate pairs of mean absolute RGB difference in [0,1]",
              "threshold_fit_on": "train", "train": summarize(train, threshold),
              "validation": summarize(validation, threshold),
              "validation_by_difficulty": {d: summarize([r for r in validation if r["difficulty"] == d], threshold)
                                           for d in ("easy", "medium", "hard")},
              "limitations": ["Predicts verdict status only; no defect type, part or evidence, so defect F1 is undefined.",
                              "The uncertain target covers clean candidates and unexposed defects alike; is_clean is not available in this export.",
                              "Scored directly on the target verdict status, not through the model output parser."]}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"threshold": threshold, "train_acc": report["train"]["status_accuracy"],
                      "validation_acc": report["validation"]["status_accuracy"]}))


if __name__ == "__main__":
    main()
