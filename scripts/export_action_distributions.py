"""Aggregate exact indistinguishable prefixes without mixing train/validation/test."""
import argparse
import json
from pathlib import Path

from turnitover.oracle.ambiguity import export_distributions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_distributions(args.source, args.out), indent=2))
