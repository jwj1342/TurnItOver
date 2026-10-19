"""Audit fixed Level 0 source data and create an asset-disjoint opaque-case split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from turnitover.oracle.level0_axis import PROMPT, read_jsonl, sha256


SOURCE_COMMIT = "f34b07c127266b401564592bfe38bb6c9e7121fb"
# Three deliberately diverse pairs train; the door and display pairs are held out.
TRAIN_ASSETS = {"PM_10697", "PM_12036", "PM_102016"}


def main(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError("Output must be fresh")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if manifest.get("version") == 2:
        from turnitover.oracle.level0_axis_v2 import prepare_v2

        prepare_v2(args.dataset, args.out, args.seed)
        return
    reviews = {r["case_id"]: r for r in read_jsonl(args.dataset / "human_review.jsonl")}
    gold = read_jsonl(args.dataset / "private" / "gold.jsonl")
    if len(reviews) != 10 or len(gold) != 10:
        raise ValueError("Expected exactly ten reviewed/gold cases")
    pairs: dict[str, list[str]] = {}
    for row in gold:
        case_id = row["case_id"]
        if case_id not in reviews or row["expected_status"] != reviews[case_id]["human_judgment"]["verdict"]:
            raise ValueError(f"Human/gold mismatch: {case_id}")
        pairs.setdefault(row["asset_id"], []).append(case_id)
    if len(pairs) != 5 or any(len(ids) != 2 for ids in pairs.values()):
        raise ValueError("Expected five clean/axis asset pairs")
    train = sorted(case for asset, cases in pairs.items() if asset in TRAIN_ASSETS for case in cases)
    heldout = sorted(case for asset, cases in pairs.items() if asset not in TRAIN_ASSETS for case in cases)
    if len(train) != 6 or len(heldout) != 4 or set(train) & set(heldout):
        raise ValueError("Invalid 3/2 asset split")
    for case_id in train + heldout:
        image = args.dataset / "inputs" / f"{case_id}.png"
        if sha256(image) != manifest["files"][f"inputs/{case_id}.png"]:
            raise ValueError(f"Image SHA mismatch: {case_id}")
        if Image.open(image).size != (542, 542):
            raise ValueError(f"Unexpected four-panel dimensions: {case_id}")
    args.out.mkdir(parents=True)
    split = {
        "source_commit": SOURCE_COMMIT,
        "task": "four-panel motion comparison -> pass|fail",
        "prompt": PROMPT,
        "train_case_ids": train,
        "heldout_case_ids": heldout,
        "train_assets": 3,
        "heldout_assets": 2,
        "asset_pair_crosses_split": False,
        "private_gold_used_only_for": "human-label equality and clean/axis pair grouping",
        "model_visible_fields": ["image", "prompt"],
        "not_model_visible": ["asset_id", "private parameters", "hidden label metadata"],
    }
    (args.out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    audit = {
        "status": "passed", "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": sha256(args.dataset / "manifest.json"),
        "human_review_sha256": sha256(args.dataset / "human_review.jsonl"),
        "private_gold_sha256": sha256(args.dataset / "private" / "gold.jsonl"),
        "input_sha256": {case: sha256(args.dataset / "inputs" / f"{case}.png") for case in train + heldout},
        "input_size": [542, 542], "native_tile_size": [256, 256],
        "human_gold_agree": True, "asset_pairs": 5, "cases": 10,
        "train_cases": len(train), "heldout_cases": len(heldout),
    }
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    main(parser.parse_args())
