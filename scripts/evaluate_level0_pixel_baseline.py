"""Transparent raw-pixel motion-delta baseline for Level 0 four-panel images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from turnitover.oracle.level0_axis import records_by_split, sha256


def score(path: Path) -> float:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    if image.shape != (542, 542, 3): raise ValueError(f"Unexpected image shape: {path}")
    # Preserve the input's row-major positions: compare reference and candidate temporal deltas.
    ref_rest, ref_limit = image[10:266, 10:266], image[10:266, 276:532]
    cand_rest, cand_limit = image[276:532, 10:266], image[276:532, 276:532]
    return float(np.mean(np.abs((ref_limit - ref_rest) - (cand_limit - cand_rest))))


def main(args: argparse.Namespace) -> None:
    if args.out.exists(): raise ValueError("Fresh output required")
    train = records_by_split(args.dataset, args.split, "train")
    heldout = records_by_split(args.dataset, args.split, "heldout")
    train_scores = [(r, score(r["image"])) for r in train]
    passes = [value for record, value in train_scores if record["label"] == "pass"]
    fails = [value for record, value in train_scores if record["label"] == "fail"]
    if not passes or not fails: raise ValueError("Balanced train labels required")
    threshold = (max(passes) + min(fails)) / 2
    args.out.mkdir(parents=True)
    rows = []
    for cohort, records in (("train", train), ("heldout", heldout)):
        for record in records:
            value = score(record["image"]); predicted = "fail" if value > threshold else "pass"
            rows.append({"cohort": cohort, "case_id": record["case_id"], "expected": record["label"], "predicted": predicted, "correct": predicted == record["label"], "motion_delta_l1": value})
    (args.out / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    metrics = {"status": "complete", "method": "mean absolute difference of reference-vs-candidate temporal RGB deltas",
               "threshold_fit": "midpoint(max train pass score, min train fail score)", "threshold": threshold,
               "split_sha256": sha256(args.split), "cohorts": {cohort: {"n": sum(r["cohort"] == cohort for r in rows), "correct": sum(r["cohort"] == cohort and r["correct"] for r in rows)} for cohort in ("train", "heldout")}}
    for item in metrics["cohorts"].values(): item["accuracy"] = item["correct"] / item["n"]
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--split", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
