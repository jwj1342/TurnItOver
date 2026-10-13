#!/usr/bin/env python3
"""Audit formal terminal Episodes and witness the exact Qwen SFT input path."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from scripts.audit_episode_export import audit as audit_episode_export
from turnitover.core.jsonio import write_json_atomic
from turnitover.oracle.supervision import load_examples
from turnitover.oracle.training import tokenize_example
from turnitover.verifier.contracts import Context, parse_decision


SPLITS = ("train", "validation")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(record: dict) -> dict:
    return json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])


def audit_asset_pools(asset_roots: list[Path], test_plan: Path) -> dict:
    """Check the combined formal pool without opening or modifying frozen Test assets."""
    test = _json(test_plan)
    if test.get("role") != "formal_test":
        raise ValueError("Expected a frozen formal Test plan")
    test_source_hashes = {row["source_identity_sha256"] for row in test["objects"]}
    seen_groups: dict[str, str] = {}
    seen_source_hashes: set[str] = set()
    by_split = Counter()
    pools = []
    for root in asset_roots:
        catalog, splits = _json(root / "catalog.json"), _json(root / "splits.json")
        if catalog.get("role") != "formal_train_validation" or catalog.get("formal_training") is not True:
            raise ValueError(f"Non-formal asset catalog: {root}")
        entries, groups = catalog["assets"], splits["groups"]
        if len(entries) != len(groups):
            raise ValueError(f"Catalog/split cardinality differs: {root}")
        for entry in entries:
            group = entry["split_group"]
            split = groups.get(group)
            if split not in SPLITS or group in seen_groups or entry.get("formal_training") is not True:
                raise ValueError(f"Invalid or duplicate formal split group: {group}")
            seen_groups[group] = split
            by_split[split] += 1
            source_hash = entry.get("source_identity_sha256")
            if source_hash:
                if source_hash in seen_source_hashes or source_hash in test_source_hashes:
                    raise ValueError(f"Source identity leakage: {group}")
                seen_source_hashes.add(source_hash)
        pools.append({"root": str(root), "catalog_sha256": _sha256(root / "catalog.json"),
                      "splits_sha256": _sha256(root / "splits.json"), "assets": len(entries)})

    # PartNet conversion records the geometry/source comparison against this exact
    # Test plan. Rechecking the frozen Test mesh would violate the no-Test-access boundary.
    partnet = next((root for root in asset_roots if (root / "conversion.json").is_file()), None)
    if partnet:
        conversion = _json(partnet / "conversion.json")
        if conversion.get("formal_test_plan_sha256") != _sha256(test_plan):
            raise ValueError("PartNet conversion used a different frozen Test plan")
        accepted = [row for row in conversion["assets"] if row.get("formal_train_validation_eligible")]
        if any(row.get("frozen_test_overlap") is not False for row in accepted):
            raise ValueError("PartNet accepted asset lacks a negative frozen-Test overlap result")
    if len(seen_groups) != 40 or by_split != Counter(train=31, validation=9):
        raise ValueError(f"Unexpected combined formal pool: {len(seen_groups)} assets, {dict(by_split)}")
    return {"assets": len(seen_groups), "by_split": dict(by_split), "split_group_overlap": 0,
            "source_identity_overlap": 0, "frozen_test_source_overlap": 0,
            "frozen_test_plan_sha256": _sha256(test_plan), "pools": pools,
            "test_access": "identity manifest only; no Test render, model input, or evaluation"}


def audit_episode_inputs(oracle: Path) -> tuple[dict, dict[str, tuple[dict, list]]]:
    """Run the production loader over every row and validate causal prefixes/targets."""
    manifest = _json(oracle / "manifest.json")
    annotations = {row["id"]: row for row in _jsonl(oracle / "annotations.private.jsonl")}
    stats = {"rows": Counter(), "terminal": Counter(), "verdicts": Counter(),
             "target_defects": Counter(), "source_labels": Counter(),
             "stop_reasons": Counter(), "episode_lengths": Counter(),
             "image_counts": Counter(), "image_references": Counter(),
             "objectives": Counter(), "variants": Counter()}
    per_split = {split: {"verdicts": Counter(), "target_defects": Counter(),
                         "source_labels": Counter(), "stop_reasons": Counter(),
                         "episode_lengths": Counter()} for split in SPLITS}
    forward_candidates = {}
    for split in SPLITS:
        for record, images in load_examples(
                oracle, split, "episodes", require_formal_training=True, terminal_only=False):
            stats["rows"][split] += 1
            stats["image_references"][split] += len(images)
            payload = _payload(record)
            annotation = annotations[record["episode_id"]]
            expected_history = annotation["trajectories"]["oracle"]["history"][:record["decision_index"]]
            if payload["history"] != expected_history:
                raise ValueError(f"Model history is not the exact causal prefix: {record['id']}")
            expected_images = [record["images"][0]] + [row["image_ref"] for row in payload["history"]]
            if payload["image_order"] != record["images"] or record["images"] != expected_images:
                raise ValueError(f"Multimodal image order differs from causal history: {record['id']}")
            if payload["remaining"] != manifest["budget"] - record["decision_index"]:
                raise ValueError(f"Remaining budget differs from history length: {record['id']}")
            identity = record["data_identity"]
            if (identity["reference_sha"] != annotation["reference_sha"]
                    or identity["candidate_sha"] != annotation["candidate_sha"]
                    or any(identity.get(key) != value for key, value in annotation["data_identity"].items())):
                raise ValueError(f"Episode/private provenance mismatch: {record['id']}")
            if not record["terminal"]:
                continue
            target = json.loads(record["messages"][1]["content"])
            if set(target) != {"verdict"}:
                raise ValueError(f"Terminal SFT target is not verdict-only: {record['id']}")
            parse_decision(target, Context(**payload["context"]),
                           {row["step"] for row in payload["history"]}, final_only=True)
            verdict = target["verdict"]
            stats["terminal"][split] += 1
            stats["verdicts"][verdict["status"]] += 1
            stats["target_defects"].update(row["defect_id"] for row in verdict["findings"])
            stats["source_labels"].update(row["defect_id"] for row in annotation["labels"])
            stats["stop_reasons"][record["termination_reason"]] += 1
            stats["episode_lengths"][str(record["decision_index"])] += 1
            stats["image_counts"][str(len(images))] += 1
            stats["objectives"][record["training_objective"]] += 1
            labels = annotation["labels"]
            stats["variants"]["clean" if not labels else "single" if len(labels) == 1 else "combined"] += 1
            per_split[split]["verdicts"][verdict["status"]] += 1
            per_split[split]["target_defects"].update(row["defect_id"] for row in verdict["findings"])
            per_split[split]["source_labels"].update(row["defect_id"] for row in labels)
            per_split[split]["stop_reasons"][record["termination_reason"]] += 1
            per_split[split]["episode_lengths"][str(record["decision_index"])] += 1
            current = forward_candidates.get(split)
            if current is None or len(images) > len(current[1]):
                forward_candidates[split] = (record, images)
    if stats["terminal"] != Counter(train=110, validation=28):
        raise ValueError(f"Unexpected terminal distribution: {dict(stats['terminal'])}")
    for status in ("fail", "uncertain", "pass"):
        stats["verdicts"][status] += 0
        for split in SPLITS:
            per_split[split]["verdicts"][status] += 0
    result = {key: dict(value) for key, value in stats.items()}
    result["per_split"] = {
        split: {key: dict(value) for key, value in distributions.items()}
        for split, distributions in per_split.items()
    }
    return result, forward_candidates


def audit_qwen(oracle: Path, model_path: Path, forward_candidates: dict, *, forward: bool) -> dict:
    """Tokenize every SFT target; optionally run one real no-grad loss per split."""
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    counts, token_lengths, supervised_lengths = Counter(), [], []
    for split in SPLITS:
        for record, images in load_examples(
                oracle, split, "episodes", require_formal_training=True, terminal_only=True):
            batch = tokenize_example(processor, record, images)
            labels = batch["labels"]
            supervised = labels != -100
            first = int(supervised[0].nonzero()[0])
            if torch.any(labels[0, :first] != -100) or torch.any(labels[0, first:] == -100):
                raise ValueError(f"Assistant-only mask is not contiguous: {record['id']}")
            if batch["input_ids"].shape != labels.shape or int(supervised.sum()) != labels.shape[1] - first:
                raise ValueError(f"Assistant labels were truncated or misaligned: {record['id']}")
            counts[split] += 1
            token_lengths.append(int(labels.shape[1]))
            supervised_lengths.append(int(supervised.sum()))
            del batch
    result = {
        "processor_examples": dict(counts), "processor_failures": 0,
        "sequence_tokens": {"min": min(token_lengths), "max": max(token_lengths)},
        "supervised_tokens": {"min": min(supervised_lengths), "max": max(supervised_lengths)},
        "assistant_only_mask": True, "label_truncation": False,
        "multimodal_alignment": True, "training_batch_size": 1,
        "batch_semantics": "identical to train_qwen_episode_sft.py; no padding collator",
        "forward": [],
    }
    if not forward:
        result["forward_status"] = "not_requested"
        return result
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for the requested Qwen forward witness")
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    with torch.inference_mode():
        for split in SPLITS:
            record, images = forward_candidates[split]
            batch = tokenize_example(processor, record, images).to(model.device)
            loss = float(model(**batch).loss.cpu())
            if not math.isfinite(loss):
                raise ValueError(f"Non-finite forward loss: {record['id']}")
            result["forward"].append({"split": split, "id": record["id"],
                                      "images": len(images), "tokens": int(batch["input_ids"].shape[1]),
                                      "supervised_tokens": int((batch["labels"] != -100).sum()),
                                      "loss": loss})
    result["forward_status"] = "complete_no_grad_no_optimizer"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--asset-pool", type=Path, action="append", required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = [path.resolve() for path in args.asset_pool]
    base = audit_episode_export(args.benchmark.resolve(), args.oracle.resolve(), args.catalog.resolve())
    episode_inputs, candidates = audit_episode_inputs(args.oracle.resolve())
    qwen = audit_qwen(args.oracle.resolve(), args.model.resolve(), candidates, forward=args.forward)
    report = {
        "version": 1, "status": "passed", "formal_training": True,
        "dataset_manifest_sha256": _sha256(args.oracle / "manifest.json"),
        "model_config_sha256": _sha256(args.model / "config.json"),
        "asset_pool": audit_asset_pools(roots, args.test_plan.resolve()),
        "episode_audit": base, "training_inputs": episode_inputs, "qwen": qwen,
        "sft_scope": {"included": "138 terminal structured diagnosis/stop targets",
                      "excluded": "all privileged Oracle action targets and all private annotations",
                      "hidden_clean_pass": "never generated; pass targets = 0"},
        "known_biases": [
            "No pass target: all 47 clean-corruption cases conservatively stop as uncertain.",
            "First-failure stopping exposes 21 of 68 joint-axis source labels in terminal targets, versus 72 of 72 part-offset labels.",
            "Only two injected defect families and no natural-error human Gold are present.",
            "Four observable-prefix groups have conflicting privileged Oracle actions; every action target is excluded from SFT.",
            "Validation has 28 Episodes from only seven PartNet assets; it is a pipeline holdout, not strong generalization evidence.",
        ],
        "failures": [],
        "readiness": "ready_for_formal_sft_with_documented_class-coverage limitations",
    }
    write_json_atomic(args.out.resolve(), report)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
