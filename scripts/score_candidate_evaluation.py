"""Score native candidate predictions without the original dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turnitover.oracle.candidate_scoring import score_candidate_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = score_candidate_snapshot(args.snapshot, args.predictions, args.out)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
