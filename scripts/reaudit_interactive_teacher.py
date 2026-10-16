"""Re-run the independent audit on saved interactive-teacher Episodes without any model call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from turnitover.teacher.audit import audit_episode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/benchmarks/partnet-mobility-formal-v1/prepared"
DEFAULT_ANNOTATIONS = (ROOT / "data/benchmarks/partnet-mobility-formal-v1/"
                       "oracle-first-failure-v4/annotations.private.jsonl")


def _rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--oracle-annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--manifest-root", type=Path,
                        help="Superseding frozen witness directory; default is the manifest the Episode recorded")
    parser.add_argument("--audit-name", default="audit.json",
                        help="Write the re-audit beside, not over, the original audit")
    parser.add_argument("--diagnosis-objective", choices=("first_failure", "comprehensive_within_budget"),
                        default="first_failure")
    args = parser.parse_args()
    gold = {**_rows(args.benchmark / "train.gold.jsonl"), **_rows(args.benchmark / "validation.gold.jsonl")}
    annotations = _rows(args.oracle_annotations)
    for source in args.episode:
        episode = json.loads((source / "episode.json").read_text())
        uid = episode["episode_id"]
        path = (args.manifest_root / uid / "manifest.json" if args.manifest_root
                else Path(episode["frozen_witness_manifest"]["path"]))
        audit = audit_episode(source, gold[uid], annotations.get(uid), manifest=json.loads(path.read_text()),
                              audit_name=args.audit_name, diagnosis_objective=args.diagnosis_objective)
        evidence = audit["independent_evidence_audit"]
        print(json.dumps({"episode": str(source), "accepted": evidence["accepted"],
                          "outcome": evidence["diagnosis_outcome"],
                          "standard": evidence.get("evidence_standard"),
                          "reasons": evidence["rejection_reasons"]}))
