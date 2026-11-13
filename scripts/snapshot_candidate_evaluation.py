"""Freeze native release observations for later inference and offline scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turnitover.oracle.candidate_snapshot import prepare_candidate_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    report = prepare_candidate_snapshot(args.dataset, args.out, args.split, manifest_sha256=args.manifest_sha256)
    print(json.dumps({key: report[key] for key in ("status", "split", "case_count", "label_scope")}))


if __name__ == "__main__":
    main()
