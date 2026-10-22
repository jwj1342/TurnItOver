#!/usr/bin/env python3
"""Read-only audit (issue #9): can simple rules on the candidate code alone separate defects from clean samples?

``extract`` reads the candidate specs of benchmark/v1, easy-batch-v1 before the fix (the preserved
``easy-batch-v1-specs-prefix``) and the fixed batch, and writes only their per-joint code features into the audit
directory, so the numbers stay reproducible from Git after the large spec files are gone. ``report`` scores the rules
on those features. No dataset file is modified and no model is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from turnitover.benchmark.code_shortcuts import FLAG_THRESHOLD, audit_dataset, joint_features
from turnitover.benchmark.manifest import load_gold, load_manifest, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/evidence/code-shortcut-audit-v1"
V1_BENCH = ROOT / "benchmark/v1"
BATCH_V1 = ROOT / "docs/evidence/easy-batch-v1"
BATCH_V2 = ROOT / "docs/evidence/easy-batch-v2"
SPECS_PREFIX = ROOT / "data/benchmarks/easy-batch-v1-specs-prefix"
SPECS_V1 = ROOT / "data/benchmarks/easy-batch-v1-specs"
SPECS_V2 = ROOT / "data/benchmarks/easy-batch-v2-specs"
RANDOM_ANGLE = "axis_random_angle"
PREFIX_STRATUM = {"joint_axis": RANDOM_ANGLE, "direction_flip": "sign_negation"}


def sample(spec_path: Path, label: str, stratum: str | None, sample_id: str) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return {"sample_id": sample_id, "label": label, "stratum": stratum, "joints": joint_features(spec)}


def extract_benchmark_v1() -> list[dict]:
    gold = load_gold(V1_BENCH)
    rows = []
    for row in load_manifest(V1_BENCH):
        defects = gold[row["sample_id"]]["defects"]
        candidate = next(c for c in row["code"] if c["role"] == "candidate_spec")
        rows.append(sample(ROOT / candidate["path"], defects[0]["defect_id"] if defects else "clean",
                           RANDOM_ANGLE if defects else None, row["sample_id"]))
    return rows


def batch_gold() -> list[dict]:
    return read_jsonl(BATCH_V1 / "private/gold.jsonl")


def extract_batch_prefix() -> list[dict]:
    return [sample(SPECS_PREFIX / f"{g['case_id']}.json", g["defect_id"] or "clean", PREFIX_STRATUM.get(g["kind"]),
                   g["case_id"]) for g in batch_gold()]


def extract_batch_fixed() -> list[dict]:
    rows = [sample(SPECS_V1 / f"{g['case_id']}.json", "clean", None, g["case_id"])
            for g in batch_gold() if g["kind"] == "clean"]
    rows += [sample(SPECS_V1 / f"{g['case_id']}.json", g["defect_id"], g["stratum"], g["case_id"])
             for g in batch_gold() if g["kind"] == "direction_flip"]
    rows += [sample(SPECS_V2 / f"{g['case_id']}.json", g["defect_id"], g["stratum"], g["case_id"])
             for g in read_jsonl(BATCH_V2 / "private/gold.jsonl")]
    return rows


DATASETS = {"benchmark-v1": extract_benchmark_v1, "easy-batch-v1-before-fix": extract_batch_prefix,
            "easy-batch-after-fix": extract_batch_fixed}


def report(features: dict) -> dict:
    return {"metric": "balanced accuracy of rules on candidate code only (0.5 = chance)",
            "flag_threshold": FLAG_THRESHOLD, "datasets": {name: audit_dataset(rows) for name, rows in features.items()}}


def render(result: dict) -> str:
    lines = []
    for name, strata in result["datasets"].items():
        for stratum, r in strata.items():
            rules = ", ".join(f"{k} {v['balanced_accuracy']:.2f}{'*' if v['above_chance'] else ''}"
                              for k, v in r["rules"].items())
            lines.append(f"{name:26s} {stratum:48s} n={r['n_defect']}/{r['n_clean']} best={r['best_rule']} "
                         f"{r['best_balanced_accuracy']:.2f} | {rules}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "report", "all"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command in ("extract", "all"):
        features = {name: extract() for name, extract in DATASETS.items()}
        (OUT / "features.json").write_text(json.dumps(features, sort_keys=True), encoding="utf-8")
    if args.command in ("report", "all"):
        result = report(json.loads((OUT / "features.json").read_text(encoding="utf-8")))
        (OUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(render(result))


if __name__ == "__main__":
    main()
