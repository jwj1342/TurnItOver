"""Audit a prepared benchmark and its Episode export without model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from turnitover.assets.catalog import CatalogSource


SPLITS = ("train", "validation", "test")
PRIVATE_INPUT_KEYS = {
    "candidate_program", "candidate_sha", "exposures", "labels", "privileged",
    "program_sha", "reference_program", "reference_sha", "signals",
}
MODEL_INPUT_KEYS = {"context", "remaining", "history", "image_order", "taxonomy_ids"}


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
    }
    split_ids, split_groups, split_references = {}, {}, {}
    image_paths: set[Path] = set()
    episode_rows = defaultdict(list)

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
            if len(payload["history"]) != row["decision_index"] or len(row["images"]) != 1 + len(payload["history"]):
                raise ValueError(f"Image/action history mismatch: {row['id']}")
            for image in row["images"]:
                path = oracle / image
                if not path.is_file() or not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError(f"Missing or invalid PNG: {path}")
                image_paths.add(path)

        final = terminal[0]
        verdict = json.loads(final["messages"][1]["content"])["verdict"]
        report["terminal_verdicts"][verdict["status"]] += 1
        report["evidence"][final["evidence"]["status"]] += 1
        trajectory = annotation["trajectories"]["oracle"]
        for finding, label_index in zip(verdict["findings"], final["evidence"]["covered_label_indices"], strict=True):
            expected_steps = [observation["step"] for observation, witness in zip(
                trajectory["history"], trajectory["witness_indices"], strict=True
            ) if label_index in annotation["exposures"][witness]]
            if finding["evidence_steps"] != expected_steps:
                raise ValueError(f"Finding evidence steps do not match private exposure audit: {uid}")
        expected_status = "pass" if not labels else "fail" if verdict["findings"] else "uncertain"
        if verdict["status"] != expected_status:
            raise ValueError(f"Terminal verdict contradicts evidence-qualified labels: {uid}")

    if episode_rows:
        raise ValueError(f"Episode rows without private annotations: {sorted(episode_rows)}")
    report.update(
        status="complete",
        samples=dict(report["samples"]), strata=dict(report["strata"]),
        quality=dict(report["quality"]), terminal_verdicts=dict(report["terminal_verdicts"]),
        evidence=dict(report["evidence"]), action_steps=dict(report["action_steps"]), episode_rows=sum(
            len(_rows(oracle / f"{split}.episodes.jsonl")) for split in SPLITS),
        referenced_pngs=len(image_paths),
        referenced_png_sha256=len({hashlib.sha256(path.read_bytes()).hexdigest() for path in image_paths}),
        split_isolation="IDs, split groups, source assets, and reference SHAs are disjoint",
        private_ground_truth="No private keys occur in model-facing JSON payloads; evidence remains row metadata",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.benchmark, args.oracle, args.catalog)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
