#!/usr/bin/env python3
"""Freeze, audit and score Benchmark v1 (milestone 1: data and protocol only, no model runs)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from turnitover.benchmark.audit import audit
from turnitover.benchmark.manifest import OUT, freeze
from turnitover.benchmark.protocol import CONDITIONS
from turnitover.benchmark.scoring import score
from turnitover.core.jsonio import write_json_atomic

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="write benchmark/v1 from the Level 0 evidence package (refuses to overwrite)")
    audit_p = sub.add_parser("audit", help="verify the frozen files and write audit.json")
    audit_p.add_argument("--out", type=Path, default=ROOT / OUT / "audit.json")
    score_p = sub.add_parser("score", help="score a predictions JSONL against frozen gold")
    score_p.add_argument("--predictions", type=Path, required=True)
    score_p.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    score_p.add_argument("--split", choices=("train", "validation"), default="validation")
    score_p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    bench = ROOT / OUT
    if args.command == "freeze":
        print(json.dumps(freeze(ROOT, bench), indent=2))
    elif args.command == "audit":
        report = audit(ROOT, bench)
        write_json_atomic(args.out, report)
        print(f"audit {report['status']}: {len(report['failures'])} failures, {len(report['gaps'])} gaps")
        for item in report["failures"] + report["gaps"]:
            print(" -", item)
        return 0 if report["status"] == "passed" else 1
    else:
        report = score(bench, args.predictions, args.out, condition=args.condition, split=args.split)
        print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
