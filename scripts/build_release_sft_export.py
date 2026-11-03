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
    args = parser.parse_args()
    print(json.dumps(export_releases(args.release, args.output, args.dataset_version,
                                     args.asset_root), indent=2))


if __name__ == "__main__":
    main()
