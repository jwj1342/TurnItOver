"""Repair Success with case-level bootstrap intervals and paired differences against reference judges.

Each judge has several runs per case (generator repetitions for the fixed judges, training seeds for ours_*). A case's
score is the mean success over that judge's runs; missing, incomplete and generator_error episodes count as failures.
Cases are resampled with replacement, the same resample for every judge, so differences between judges are paired by
case (object). Usage:
  python scripts/exp900/analysis/repair_bootstrap.py --gen Qwen3.8-27B-FP8 --cases tio-runs/repair/eligible-v11.txt \
      --runs tio-runs/repair/q38/pilot:none,pixel,gold tio-runs/repair/q38/rep2 ... [--ref none pixel] [--samples 10000]
A run argument is <run root>[:<judge>,<judge>...]; without a judge list every judge directory under the root is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ORDER = ("none", "pixel", "zs8b_screenshot", "zs8b_sweep", "zsapi_sweep", "ours_2b_ordinary", "ours_2b_evidence",
         "ours_8b_evidence", "gold")


def success(path: Path) -> float:
    if not path.exists():
        return 0.0
    result = json.loads(path.read_text())
    return float(result["status"] == "complete" and bool(result["final_passed"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--ref", nargs="+", default=["none", "pixel"])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    cases = [l.strip() for l in args.cases.read_text().splitlines() if l.strip() and not l.startswith("#")]
    runs: dict[str, list[list[float]]] = {}
    for spec in args.runs:
        root, _, only = spec.partition(":")
        judges = only.split(",") if only else sorted(p.name for p in Path(root).iterdir() if (p / args.gen).is_dir())
        for judge in judges:
            runs.setdefault(judge, []).append([success(Path(root) / judge / args.gen / judge / c / "result.json") for c in cases])
    judges = [j for j in ORDER if j in runs] + sorted(set(runs) - set(ORDER))
    score = {j: np.array(runs[j]).mean(axis=0) for j in judges}          # per case, mean over runs
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(cases), size=(args.samples, len(cases)))
    boot = {j: score[j][idx].mean(axis=1) for j in judges}
    ci = lambda v: (100 * np.percentile(v, 2.5), 100 * np.percentile(v, 97.5))
    head = "| Judge | runs | Success | 95% CI | " + " | ".join(f"diff vs {r} [95% CI]" for r in args.ref) + " |"
    lines = [f"# Repair Success, {len(cases)} cases, generator {args.gen}; case-level bootstrap ({args.samples} resamples, "
             f"seed {args.seed}), paired by case", "", head, "|---|--:|--:|--:|" + "--:|" * len(args.ref)]
    for j in judges:
        lo, hi = ci(boot[j])
        diffs = []
        for r in args.ref:
            if r not in score or r == j:
                diffs.append("--")
                continue
            d = boot[j] - boot[r]
            dlo, dhi = ci(d)
            diffs.append(f"{100 * (score[j].mean() - score[r].mean()):+.1f} [{dlo:+.1f}, {dhi:+.1f}]")
        lines.append(f"| {j} | {len(runs[j])} | {100 * score[j].mean():.1f} | {lo:.1f}-{hi:.1f} | " + " | ".join(diffs) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
