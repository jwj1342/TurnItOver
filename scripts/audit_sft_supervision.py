#!/usr/bin/env python3
"""Audit SFT targets for visible-prefix leakage, consistency, and target drift."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic
from turnitover.oracle.sft_data import load_terminal_training_examples


FORBIDDEN_ASSISTANT_TERMS = ("injected", "private label", "hidden defect", "unexposed defect")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _findings(record: dict) -> set[tuple[str, str | None, tuple[str, ...]]]:
    verdict = json.loads(record["messages"][1]["content"])["verdict"]
    return {(row["defect_id"], row["joint_id"], tuple(sorted(row["parts"])))
            for row in verdict["findings"]}


def _public_key(record: dict) -> str:
    public = {"user": record["messages"][0]["content"], "images": record["image_sha256"]}
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def _payload(record: dict) -> dict:
    return json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])


def _target_changes(before: Path, after: Path) -> dict:
    counts = Counter()
    examples = []
    for split in ("train", "validation"):
        old = {row["id"]: row for row in _rows(before / f"{split}.episodes.jsonl")}
        new = {row["id"]: row for row in _rows(after / f"{split}.episodes.jsonl")}
        if set(old) != set(new):
            raise ValueError(f"Episode roster changed during supervision-only re-export: {split}")
        for uid in sorted(old):
            left, right = old[uid], new[uid]
            if (left["messages"][0] != right["messages"][0]
                    or left["image_sha256"] != right["image_sha256"]):
                raise ValueError(f"Public Episode input changed: {uid}")
            if left["messages"][1] == right["messages"][1]:
                counts["unchanged_rows"] += 1
                continue
            counts["changed_rows"] += 1
            if not left["terminal"] or not right["terminal"]:
                raise ValueError(f"Non-terminal action target changed: {uid}")
            old_verdict = json.loads(left["messages"][1]["content"])["verdict"]
            new_verdict = json.loads(right["messages"][1]["content"])["verdict"]
            for field in ("status", "summary", "limitations", "findings"):
                counts[f"changed_{field}"] += old_verdict[field] != new_verdict[field]
            old_steps = [row["evidence_steps"] for row in old_verdict["findings"]]
            new_steps = [row["evidence_steps"] for row in new_verdict["findings"]]
            counts["changed_evidence_steps"] += old_steps != new_steps
            examples.append({
                "id": uid, "split": split,
                "before_summary": old_verdict["summary"],
                "after_summary": new_verdict["summary"],
                "before_limitations": old_verdict["limitations"],
                "after_limitations": new_verdict["limitations"],
            })
    return {"counts": dict(counts), "changed_examples": examples}


def _integrity(dataset: Path) -> dict:
    counts = Counter()
    public_groups = defaultdict(set)
    no_evidence_targets = set()
    leak_examples = []
    stop_feature_labels = defaultdict(set)
    for split in ("train", "validation"):
        episodes = _rows(dataset / f"{split}.episodes.jsonl")
        stopping = _rows(dataset / f"{split}.stopping.jsonl")
        episode_keys = {(row["episode_id"], row["decision_index"]) for row in episodes}
        stop_keys = {(row["episode_id"], row["decision_index"]) for row in stopping}
        if episode_keys != stop_keys:
            raise ValueError(f"Episode and stop prefix rosters differ: {split}")
        for record in episodes:
            counts["episode_rows"] += 1
            assistant = record["messages"][1]["content"]
            public_groups[_public_key(record)].add(assistant)
            lowered = assistant.lower()
            matched = [term for term in FORBIDDEN_ASSISTANT_TERMS if term in lowered]
            if matched:
                counts["leaking_rows"] += 1
                leak_examples.append({"id": record["id"], "terms": matched})
            if not record["terminal"]:
                counts["action_rows"] += 1
                continue
            counts["terminal_rows"] += 1
            verdict = json.loads(assistant)["verdict"]
            history_steps = {row["step"] for row in _payload(record)["history"]}
            for finding in verdict["findings"]:
                if not set(finding["evidence_steps"]) <= history_steps:
                    raise ValueError(f"Finding cites a future or unknown step: {record['id']}")
            if not record["evidence"]["covered_label_indices"]:
                no_evidence_targets.add(assistant)
        for record in stopping:
            counts["stopping_rows"] += 1
            payload = _payload(record)
            decision = record["stop_target"]
            outcome = record["stop_outcome"]
            counts[f"stop:{decision}:{outcome or 'none'}"] += 1
            key = (record["decision_index"], payload["remaining"], len(payload["history"]),
                   len(record["images"]))
            stop_feature_labels[key].add(decision)
            has_evidence = bool(record["evidence"]["covered_label_indices"])
            if decision == "continue" and (record["terminal"] or payload["remaining"] <= 0):
                raise ValueError(f"Invalid continue target: {record['id']}")
            if outcome == "fail" and not has_evidence:
                raise ValueError(f"Evidence-free fail stop: {record['id']}")
            if outcome == "uncertain" and has_evidence:
                raise ValueError(f"Evidence-backed uncertain stop: {record['id']}")
    conflicts = {key: sorted(value) for key, value in public_groups.items() if len(value) > 1}
    budget_conflicts = {str(key): sorted(value) for key, value in stop_feature_labels.items()
                        if len(value) > 1}
    return {
        "counts": dict(counts),
        "assistant_leak_terms": list(FORBIDDEN_ASSISTANT_TERMS),
        "assistant_leak_examples": leak_examples,
        "exact_public_prefix_groups": len(public_groups),
        "exact_public_prefix_conflicts": conflicts,
        "no_evidence_terminal_target_variants": len(no_evidence_targets),
        "stop_target_predictable_from_budget_features_only": not budget_conflicts,
        "stop_budget_feature_conflicts": budget_conflicts,
    }


def audit(dataset: Path, baseline_dataset: Path | None = None) -> dict:
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
    result = {
        "version": 2, "status": "complete", "split": "train",
        "terminal_prefixes_identical": True, "model_inputs_identical": True,
        "private_oracle_inputs": False, "privileged_action_targets": False,
        "counts": dict(counts), "ordinary_only_findings": examples,
        "interpretation": (
            "ordinary-only findings are Gold targets unavailable from the frozen visible evidence"),
        "visible_prefix_integrity": _integrity(dataset),
    }
    if baseline_dataset is not None:
        result["target_changes"] = _target_changes(baseline_dataset, dataset)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset.resolve(), args.baseline_dataset.resolve() if args.baseline_dataset else None)
    write_json_atomic(args.out.resolve(), report)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
