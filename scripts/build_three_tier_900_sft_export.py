#!/usr/bin/env python3
"""Build the frozen 900-case Easy/Medium/Hard Episode SFT export."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turnitover.oracle.release_sft_export import export_three_tier_900


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="append", type=Path, required=True,
                        help="Easy and Medium release directories")
    parser.add_argument("--hard-candidate-zip", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True,
                        help="JSON mapping source_dataset:source_asset_id to train|validation|test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--asset-root", action="append", type=Path, required=True)
    args = parser.parse_args()
    split_plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
    split_plan = split_plan.get("source_splits", split_plan) if isinstance(split_plan, dict) else split_plan
    if not isinstance(split_plan, dict):
        raise ValueError("split plan must be a JSON object")
    print(json.dumps(export_three_tier_900(args.release, args.hard_candidate_zip, args.output,
                                            args.dataset_version, args.asset_root, split_plan), indent=2))


if __name__ == "__main__":
    main()
