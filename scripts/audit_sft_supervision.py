#!/usr/bin/env python3
"""Compare ordinary and visible-evidence terminal supervision on identical inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic
from turnitover.oracle.sft_data import load_terminal_training_examples


def _findings(record: dict) -> set[tuple[str, str | None, tuple[str, ...]]]:
    verdict = json.loads(record["messages"][1]["content"])["verdict"]
    return {(row["defect_id"], row["joint_id"], tuple(sorted(row["parts"])))
            for row in verdict["findings"]}


def audit(dataset: Path) -> dict:
    ordinary = load_terminal_training_examples(dataset, "train", "ordinary")
    evidence = load_terminal_training_examples(dataset, "train", "evidence")
    counts = Counter()
    examples = []
    for (ordinary_record, _), (evidence_record, _) in zip(ordinary, evidence, strict=True):
        if (ordinary_record["id"] != evidence_record["id"]
                or ordinary_record["images"] != evidence_record["images"]
                or ordinary_record["messages"][0] != evidence_record["messages"][0]
                or ordinary_record["decision_index"] != evidence_record["decision_index"]):
            raise ValueError("Controlled supervision inputs or terminal prefixes differ")
        ordinary_findings, evidence_findings = _findings(ordinary_record), _findings(evidence_record)
        if not evidence_findings <= ordinary_findings:
            raise ValueError(f"Evidence target is not a subset of Gold: {ordinary_record['id']}")
        beyond = ordinary_findings - evidence_findings
        counts["examples"] += 1
        counts["ordinary_findings"] += len(ordinary_findings)
        counts["evidence_findings"] += len(evidence_findings)
        counts["beyond_visible_evidence"] += len(beyond)
        counts["examples_with_beyond_visible_evidence"] += bool(beyond)
        if beyond:
            examples.append({"id": ordinary_record["id"],
                             "asset_id": ordinary_record["data_identity"]["asset_id"],
                             "beyond_visible_evidence": [list(row) for row in sorted(beyond)]})
    if counts["examples"] != 110:
        raise ValueError(f"Expected 110 controlled Train examples, got {counts['examples']}")
    return {"version": 1, "status": "complete", "split": "train",
            "terminal_prefixes_identical": True, "model_inputs_identical": True,
            "private_oracle_inputs": False, "privileged_action_targets": False,
            "counts": dict(counts), "ordinary_only_findings": examples,
            "interpretation": "ordinary-only findings are Gold targets unavailable from the frozen visible evidence"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset.resolve())
    write_json_atomic(args.out.resolve(), report)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
