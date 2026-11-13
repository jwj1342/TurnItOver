#!/usr/bin/env python3
"""Build a PR #13-compatible SFT export from release directories."""
import argparse
import json
from pathlib import Path

from turnitover.oracle.release_sft_export import export_releases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--asset-root", action="append", type=Path, required=True)
    parser.add_argument("--easy-board-root", action="append", type=Path, default=[],
                        help="Native Easy boards used to recover verified mid frames")
    parser.add_argument("--exposure-audit", type=Path,
                        help="Hash-bound per-label paired exposure audit for composite cases")
    args = parser.parse_args()
    exposure_audits = json.loads(args.exposure_audit.read_text()) if args.exposure_audit else None
    print(json.dumps(export_releases(args.release, args.output, args.dataset_version,
                                     args.asset_root, easy_board_roots=args.easy_board_root,
                                     exposure_audits=exposure_audits), indent=2))


if __name__ == "__main__":
    main()
