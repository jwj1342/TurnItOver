"""Eligible repair cases from the two scripted calibration runs (no model involved).

A case is eligible when the independent audit sees its defect (scripted-stop: the untouched initial program fails)
and the audit accepts a perfect repair (scripted-fix: the program replaced by the reference passes).
Every judge in the repair table is then scored on this same set.

Usage: python scripts/exp900/eligible_cases.py --stop <OUT>/scripted-stop/none --fix <OUT>/scripted-fix/none --write eligible.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def outcomes(judge_dir: Path) -> dict[str, bool]:
    rows = {}
    for line in (judge_dir / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["case"]] = bool(row["final_passed"])     # later rows of a resumed run win
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stop", type=Path, required=True, help="<out>/scripted-stop/<judge> of the scripted-stop run")
    parser.add_argument("--fix", type=Path, required=True, help="<out>/scripted-fix/<judge> of the scripted-fix run")
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args()
    stop, fix = outcomes(args.stop), outcomes(args.fix)
    cases = sorted(set(stop) | set(fix))
    eligible = [c for c in cases if c in stop and c in fix and not stop[c] and fix[c]]
    lines = [f"# eligible {len(eligible)} of {len(cases)}"]
    for case in cases:
        if case not in eligible:
            reason = ("missing run" if case not in stop or case not in fix
                      else "audit passes the untouched candidate" if stop[case] else "audit rejects the reference")
            lines.append(f"# excluded {case}: {reason}")
    args.write.write_text("\n".join(lines + eligible) + "\n")
    print("\n".join(lines[:1] + [l for l in lines[1:]]))


if __name__ == "__main__":
    main()
