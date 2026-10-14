#!/usr/bin/env python3
"""Run Qwen on real causal Episode prefixes for shortcut diagnostics."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.oracle.supervision import load_examples
from turnitover.oracle.training import tokenize_prompt
from turnitover.verifier.contracts import Context, Verdict, parse_decision, parse_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(label: dict) -> tuple[str, str | None, tuple[str, ...]]:
    joint = label.get("params", {}).get("joint") if label["defect_id"] == "kinematics.joint_axis" else None
    return label["defect_id"], joint, tuple(sorted(label["parts"]))


def visible_findings(annotation: dict, decision_index: int) -> dict[tuple, set[int]]:
    """Private scoring view only; callers must never place this in a model message."""
    trajectory = annotation["trajectories"]["oracle"]
    history = trajectory["history"][:decision_index]
    witnesses = trajectory["witness_indices"][:decision_index]
    result: dict[tuple, set[int]] = {}
    for label_index, label in enumerate(annotation["labels"]):
        steps = {observation["step"] for observation, witness in zip(history, witnesses, strict=True)
                 if label_index in annotation["exposures"][witness]}
        if steps:
            result[_signature(label)] = steps
    return result


def _payload(record: dict) -> tuple[str, dict]:
    protocol, payload_text = record["messages"][0]["content"].rsplit("\n", 1)
    return protocol, json.loads(payload_text)


def remove_remaining(record: dict) -> dict:
    """Delete, rather than fabricate, budget metadata while preserving the real prefix."""
    record = copy.deepcopy(record)
    protocol, payload = _payload(record)
    if "remaining" not in payload:
        raise ValueError("Episode payload has no remaining field")
    del payload["remaining"]
    record["messages"][0]["content"] = protocol + "\n" + json.dumps(payload)
    return record


def _serial(signatures) -> list[list]:
    return [[defect, joint, list(parts)] for defect, joint, parts in sorted(signatures)]


def evaluate(dataset: Path, model_path: Path, output: Path, *, adapter: Path | None,
             max_new_tokens: int, condition: str) -> dict:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    if not torch.cuda.is_available():
        raise ValueError("Qwen evaluation requires CUDA")
    output.mkdir(parents=True)
    terminal_only = condition == "terminal_without_remaining"
    examples = list(load_examples(dataset, "validation", "episodes",
                                  require_formal_training=True, terminal_only=terminal_only))
    annotations = {row["id"]: row for row in map(json.loads,
        (dataset / "annotations.private.jsonl").read_text(encoding="utf-8").splitlines())
        if row["split"] == "validation"}
    processor_source = adapter if adapter and (adapter / "preprocessor_config.json").exists() else model_path
    processor = AutoProcessor.from_pretrained(processor_source, local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    started = time.perf_counter()
    valid = 0
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for source_record, images in examples:
                record = remove_remaining(source_record) if terminal_only else source_record
                _, original_payload = _payload(source_record)
                _, model_payload = _payload(record)
                context = Context(**original_payload["context"])
                observed = {row["step"] for row in original_payload["history"]}
                annotation = annotations[source_record["episode_id"]]
                expected = visible_findings(annotation, source_record["decision_index"])
                expected_status = "fail" if expected else "uncertain"
                _, inputs = tokenize_prompt(processor, record, images)
                inputs = inputs.to(model.device)
                start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                text = processor.batch_decode(generated, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
                decision = None
                error = None
                try:
                    decision = parse_decision(parse_json(text), context, observed,
                                              final_only=original_payload["remaining"] == 0)
                    valid += 1
                except ValueError as exc:
                    error = str(exc)
                if isinstance(decision, Verdict):
                    kind = "verdict"
                    status = decision.status
                    predicted = {(row.defect_id, row.joint_id, tuple(sorted(row.parts))): set(row.evidence_steps)
                                 for row in decision.findings}
                elif decision is not None:
                    kind = "action"
                    status = None
                    predicted = {}
                else:
                    kind = "invalid"
                    status = None
                    predicted = {}
                unsupported = {signature for signature, steps in predicted.items()
                               if signature not in expected or not steps <= expected.get(signature, set())}
                labels = annotation["labels"]
                source_defects = sorted({row["defect_id"] for row in labels})
                row = {
                    "id": source_record["id"], "episode_id": source_record["episode_id"],
                    "asset_id": source_record["data_identity"]["asset_id"],
                    "condition": condition, "terminal": source_record["terminal"],
                    "decision_index": source_record["decision_index"],
                    "remaining": original_payload["remaining"], "remaining_in_input": "remaining" in model_payload,
                    "history_length": len(original_payload["history"]), "image_count": len(images),
                    "evidence_status": source_record["evidence"]["status"],
                    "evidence_available": bool(expected), "termination_reason": source_record.get("termination_reason"),
                    "source_defects": source_defects,
                    "defect_combination": "+".join(source_defects) if source_defects else "clean",
                    "expected_status": expected_status, "expected_findings": _serial(expected),
                    "expected_evidence_steps": {json.dumps(list(key)): sorted(value) for key, value in expected.items()},
                    "decision_kind": kind, "predicted_status": status,
                    "predicted_findings": _serial(predicted), "unsupported_findings": len(unsupported),
                    "unsupported_signatures": _serial(unsupported), "error": error, "raw": text,
                    "public_input_sha256": hashlib.sha256(record["messages"][0]["content"].encode()).hexdigest(),
                    "image_sha256": source_record["image_sha256"],
                    "input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": int(generated.shape[1]),
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps({key: value for key, value in row.items() if key not in ("raw", "image_sha256")}), flush=True)
        report = {
            "version": 1, "status": "complete", "condition": condition, "n": len(examples),
            "valid_decisions": valid, "model": str(model_path), "adapter": str(adapter) if adapter else None,
            "model_config_sha256": _sha256(model_path / "config.json"),
            "adapter_sha256": _sha256(adapter / "adapter_model.safetensors") if adapter else None,
            "dataset": str(dataset), "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
            "validation_episodes_sha256": _sha256(dataset / "validation.episodes.jsonl"),
            "max_new_tokens": max_new_tokens, "decode": "greedy", "elapsed_seconds": time.perf_counter() - started,
            "input_intervention": "delete_remaining_key_only" if terminal_only else "none",
            "private_scoring_fields_excluded_from_model_input": ["labels", "exposures", "witness_indices", "evidence"],
        }
    except Exception as exc:
        report = {"version": 1, "status": "incomplete", "error_type": type(exc).__name__, "error": str(exc)}
        raise
    finally:
        (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", choices=("causal_prefixes", "terminal_without_remaining"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    result = evaluate(args.dataset.resolve(), args.model.resolve(), args.out.resolve(),
                      adapter=args.adapter.resolve() if args.adapter else None,
                      max_new_tokens=args.max_new_tokens, condition=args.condition)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
