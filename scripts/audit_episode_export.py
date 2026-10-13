"""Audit a prepared benchmark and its Episode export without model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from turnitover.assets.catalog import CatalogSource
from turnitover.oracle.ambiguity import observable_key
from turnitover.oracle.pass_evidence import assess_single_reference_pass


SPLITS = ("train", "validation", "test")
PRIVATE_INPUT_KEYS = {
    "candidate_program", "candidate_sha", "exposures", "labels", "privileged",
    "program_sha", "reference_program", "reference_sha", "signals",
}
MODEL_INPUT_KEYS = {"context", "remaining", "history", "image_order", "taxonomy_ids", "training_objective"}
TRAINING_OBJECTIVES = ("first_failure", "comprehensive_within_budget")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _payload(message: str) -> dict:
    try:
        return json.loads(message.rsplit("\n", 1)[-1])
    except json.JSONDecodeError as exc:
        raise ValueError("Episode user message does not end in a JSON payload") from exc


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(v) for v in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(v) for v in value), set())
    return set()


def audit(benchmark: Path, oracle: Path, catalog: Path) -> dict:
    prepared = benchmark / "prepared"
    prepared_manifest = json.loads((prepared / "manifest.json").read_text())
    oracle_manifest = json.loads((oracle / "manifest.json").read_text())
    if prepared_manifest.get("status") != "complete" or oracle_manifest.get("status") != "complete":
        raise ValueError("Complete prepared and oracle manifests required")

    allowed_assets = {spec.asset_id for spec in CatalogSource(str(catalog))}
    annotations = {row["id"]: row for row in _rows(oracle / "annotations.private.jsonl")}
    if len(annotations) != sum(oracle_manifest["samples"].values()):
        raise ValueError("Annotation/sample count mismatch")
    unexpected = {row["asset_id"] for row in annotations.values()} - allowed_assets
    if unexpected:
        raise ValueError(f"Episode export contains assets outside the reviewed catalog: {sorted(unexpected)}")

    report = {
        "version": 1,
        "status": "incomplete",
        "benchmark": str(benchmark.resolve()),
        "oracle": str(oracle.resolve()),
        "allowed_assets": sorted(allowed_assets),
        "samples": Counter(),
        "strata": Counter(),
        "quality": Counter(),
        "terminal_verdicts": Counter(),
        "evidence": Counter(),
        "action_steps": Counter(),
        "termination_reasons": Counter(),
        "training_objectives": Counter(),
        "evidence_limited_cases": [],
        "unsupported_pass_cases": [],
        "single_reference_pass_reasons": Counter(),
        "single_reference_public_facts": Counter(),
    }
    split_ids, split_groups, split_references = {}, {}, {}
    image_paths: set[Path] = set()
    episode_rows = defaultdict(list)
    observable_actions = defaultdict(list)
    observable_image_hashes = {}

    for split in SPLITS:
        inputs = _rows(prepared / f"{split}.inputs.jsonl")
        gold = _rows(prepared / f"{split}.gold.jsonl")
        episodes = _rows(oracle / f"{split}.episodes.jsonl")
        input_ids, gold_ids = {row["id"] for row in inputs}, {row["id"] for row in gold}
        if input_ids != gold_ids:
            raise ValueError(f"Prepared input/gold IDs differ in {split}")
        split_ids[split] = input_ids
        split_groups[split] = {row["split_group"] for row in gold}
        split_references[split] = {
            (row["source_id"], row["asset_id"], row["reference_sha"]) for row in gold
        }
        for row in gold:
            quality = row["quality"]
            report["quality"][quality["status"]] += 1
            if quality["status"] != "accepted" or any(
                    label["status"] != "valid" or label["isolated_status"] != "valid"
                    for label in quality["labels"]):
                raise ValueError(f"Unqualified sample entered prepared data: {row['id']}")
        for row in episodes:
            if row["episode_id"] not in input_ids:
                raise ValueError(f"Episode not present in its prepared split: {row['episode_id']}")
            episode_rows[row["episode_id"]].append(row)
            if not row["terminal"]:
                key = observable_key(row, oracle, observable_image_hashes)
                observable_actions[(split, key)].append(row)

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            if split_ids[left] & split_ids[right] or split_groups[left] & split_groups[right]:
                raise ValueError(f"ID or split-group leakage between {left} and {right}")
            left_refs = {(source, asset) for source, asset, _ in split_references[left]}
            right_refs = {(source, asset) for source, asset, _ in split_references[right]}
            left_sha = {sha for _, _, sha in split_references[left]}
            right_sha = {sha for _, _, sha in split_references[right]}
            if left_refs & right_refs or left_sha & right_sha:
                raise ValueError(f"Reference leakage between {left} and {right}")

    for uid, annotation in annotations.items():
        rows = sorted(episode_rows.pop(uid), key=lambda row: row["decision_index"])
        if not rows or [row["decision_index"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"Non-contiguous decisions: {uid}")
        terminal = [row for row in rows if row["terminal"]]
        if len(terminal) != 1 or terminal[0] is not rows[-1]:
            raise ValueError(f"Episode needs exactly one final terminal target: {uid}")
        labels = annotation["labels"]
        report["action_steps"][terminal[0]["decision_index"]] += 1
        report["samples"][annotation["split"]] += 1
        report["strata"]["clean" if not labels else "single" if len(labels) == 1 else "combined"] += 1

        for row in rows:
            if len(row["messages"]) != 2 or [m["role"] for m in row["messages"]] != ["user", "assistant"]:
                raise ValueError(f"Invalid message roles: {row['id']}")
            payload = _payload(row["messages"][0]["content"])
            if set(payload) != MODEL_INPUT_KEYS:
                raise ValueError(f"Unexpected model-input fields: {sorted(set(payload) - MODEL_INPUT_KEYS)}")
            leaked = _keys(payload) & PRIVATE_INPUT_KEYS
            if leaked:
                raise ValueError(f"Private ground-truth keys leaked into model input: {sorted(leaked)}")
            objective = row.get("training_objective")
            if objective not in TRAINING_OBJECTIVES or payload["training_objective"] != objective:
                raise ValueError(f"Missing or inconsistent model-visible training objective: {row['id']}")
            if len(payload["history"]) != row["decision_index"] or len(row["images"]) != 1 + len(payload["history"]):
                raise ValueError(f"Image/action history mismatch: {row['id']}")
            for image in row["images"]:
                path = oracle / image
                if not path.is_file() or not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError(f"Missing or invalid PNG: {path}")
                image_paths.add(path)

        final = terminal[0]
        final_payload = _payload(final["messages"][0]["content"])
        pass_assessment = assess_single_reference_pass(final_payload, oracle)
        report["single_reference_pass_reasons"][pass_assessment["reason"]] += 1
        for fact in ("front_rest_exact_match", "runtime_stats_complete"):
            if pass_assessment.get(fact):
                report["single_reference_public_facts"][fact] += 1
        verdict = json.loads(final["messages"][1]["content"])["verdict"]
        reason = final.get("termination_reason")
        objective = final["training_objective"]
        if objective != oracle_manifest.get("episode_objective"):
            raise ValueError(f"Episode objective differs from its manifest: {uid}")
        report["terminal_verdicts"][verdict["status"]] += 1
        report["evidence"][final["evidence"]["status"]] += 1
        report["termination_reasons"][reason] += 1
        report["training_objectives"][objective] += 1
        trajectory = annotation["trajectories"]["oracle"]
        for finding, label_index in zip(verdict["findings"], final["evidence"]["covered_label_indices"], strict=True):
            label = labels[label_index]
            expected_joint = label.get("params", {}).get("joint") if label["defect_id"] == "kinematics.joint_axis" else None
            if (finding["defect_id"] != label["defect_id"] or finding["parts"] != list(label["parts"])
                    or finding["joint_id"] != expected_joint):
                raise ValueError(f"Finding identity does not match its exposed label: {uid}")
            expected_steps = [observation["step"] for observation, witness in zip(
                trajectory["history"][:final["decision_index"]],
                trajectory["witness_indices"][:final["decision_index"]], strict=True
            ) if label_index in annotation["exposures"][witness]]
            if finding["evidence_steps"] != expected_steps:
                raise ValueError(f"Finding evidence steps do not match private exposure audit: {uid}")
        expected_status = (
            "fail" if verdict["findings"] else "pass" if pass_assessment["eligible"] else "uncertain"
        )
        if verdict["status"] != expected_status:
            raise ValueError(f"Terminal verdict contradicts evidence-qualified labels: {uid}")
        first_exposure = next((index + 1 for index, witness in enumerate(trajectory["witness_indices"])
                               if annotation["exposures"][witness]), None)
        exhausted_reason = ("budget_exhausted" if len(trajectory["history"]) >= oracle_manifest["budget"]
                            else "observation_space_exhausted")
        expected_reason = ("evidence_sufficient" if objective == "first_failure" and verdict["findings"]
                           else exhausted_reason)
        expected_step = first_exposure if objective == "first_failure" and verdict["findings"] else len(trajectory["history"])
        if reason != expected_reason or final["decision_index"] != expected_step:
            raise ValueError(f"Terminal reason or stopping step is inconsistent: {uid}")
        if verdict["status"] == "pass" and not pass_assessment["eligible"]:
            report["unsupported_pass_cases"].append(uid)
        if final["evidence"]["status"] != "complete":
            report["evidence_limited_cases"].append({
                "id": uid, "asset_id": annotation["asset_id"], "split": annotation["split"],
                "labels": [label["defect_id"] for label in labels],
                "evidence": final["evidence"]["status"], "verdict": verdict["status"],
                "termination_reason": reason,
            })

    if episode_rows:
        raise ValueError(f"Episode rows without private annotations: {sorted(episode_rows)}")
    conflicts = []
    for (split, key), rows in observable_actions.items():
        targets = Counter(row["messages"][1]["content"] for row in rows)
        if len(targets) > 1:
            conflicts.append({"split": split, "observable_key": key, "examples": len(rows),
                              "distinct_actions": len(targets), "ids": [row["id"] for row in rows]})
    singleton_groups = sum(len(rows) == 1 for rows in observable_actions.values())
    report.update(
        status="complete",
        samples=dict(report["samples"]), strata=dict(report["strata"]),
        quality=dict(report["quality"]), terminal_verdicts=dict(report["terminal_verdicts"]),
        evidence=dict(report["evidence"]), action_steps=dict(report["action_steps"]),
        termination_reasons=dict(report["termination_reasons"]),
        training_objectives=dict(report["training_objectives"]), episode_rows=sum(
            len(_rows(oracle / f"{split}.episodes.jsonl")) for split in SPLITS),
        referenced_pngs=len(image_paths),
        referenced_png_sha256=len({hashlib.sha256(path.read_bytes()).hexdigest() for path in image_paths}),
        split_isolation="IDs, split groups, source assets, and reference SHAs are disjoint",
        private_ground_truth="No private keys occur in model-facing JSON payloads; evidence remains row metadata",
        observable_action_audit={
            "groups": len(observable_actions),
            "source_examples": sum(len(rows) for rows in observable_actions.values()),
            "conflicting_groups": len(conflicts), "affected_examples": sum(row["examples"] for row in conflicts),
            "singleton_groups": singleton_groups,
            "non_singleton_groups": len(observable_actions) - singleton_groups,
            "conflicts": conflicts,
            "learnability": "unresolved: exact-match conflicts are absent, but singleton privileged targets remain",
        },
        teacher_policy={
            "condition": oracle_manifest.get("condition"),
            "uses_private_counterfactual_exposure": True,
            "safe_use": "Action targets require a visible-only teacher or an explicit privileged-teacher limitation.",
        },
        single_reference_pass_audit={
            "eligible": sum(count for reason, count in report["single_reference_pass_reasons"].items()
                            if reason == "complete_public_scope"),
            "reasons": dict(report["single_reference_pass_reasons"]),
            "public_facts": dict(report["single_reference_public_facts"]),
            "rule": "Only an explicitly model-visible front/rest-only or runtime-budget-only scope can pass.",
            "full_articulated_object": "unsupported without matched reference motion/views",
        },
    )
    if report["unsupported_pass_cases"]:
        raise ValueError(f"Pass targets lack a model-visible negative-evidence mechanism: {report['unsupported_pass_cases']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.benchmark, args.oracle, args.catalog)
    if args.baseline:
        before = json.loads(args.baseline.read_text())
        result["before_after"] = {
            key: {"before": before.get(key), "after": result.get(key)}
            for key in ("terminal_verdicts", "evidence", "action_steps", "termination_reasons", "episode_rows")
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
