"""Export independently accepted interactive-teacher Episodes for SFT loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from turnitover.teacher.export import export_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_episodes(args.episode, args.out), indent=2))
