"""Score full frozen UniPhys acquisition batches from a private exposure sidecar."""
from __future__ import annotations

import argparse
from pathlib import Path

from turnitover.verifier.acquisition_scoring import score_acquisition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                        help="Complete batch directory, once per policy label")
    parser.add_argument("--exposure-sidecar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        label, separator, directory = item.partition("=")
        if not separator or not label or not directory or label in runs:
            parser.error("--run must be unique LABEL=DIR")
        runs[label] = Path(directory).resolve()
    report = score_acquisition(runs, args.exposure_sidecar.resolve(), args.out.resolve())
    print(f"scored {len(report['runs'])} policies to {args.out.resolve()}")


if __name__ == "__main__":
    main()
