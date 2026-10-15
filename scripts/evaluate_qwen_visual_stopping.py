#!/usr/bin/env python3
"""Evaluate Qwen continue/stop predictions, strict pairs, and image masking."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.oracle.stopping_metrics import summarize_pairs, summarize_stopping
from turnitover.oracle.supervision import load_examples
from turnitover.oracle.training import tokenize_prompt
from turnitover.verifier.contracts import parse_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _parse(text: str) -> tuple[str, str | None]:
    value = parse_json(text)
    if set(value) != {"stop_decision"} or not isinstance(value["stop_decision"], dict):
        raise ValueError("Expected one stop_decision object")
    decision = value["stop_decision"].get("decision")
    outcome = value["stop_decision"].get("outcome")
    if decision not in ("continue", "stop"):
        raise ValueError("Invalid stop decision")
    if decision == "continue" and outcome is not None:
        raise ValueError("Continue must have null outcome")
    if decision == "stop" and outcome not in ("fail", "uncertain"):
        raise ValueError("Stop requires fail or uncertain outcome")
    return decision, outcome


def evaluate(dataset: Path, model_path: Path, output: Path, *, adapter: Path | None,
             max_new_tokens: int, image_mode: str) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    if image_mode not in ("original", "masked"):
        raise ValueError(f"Unknown image mode: {image_mode}")
    if not torch.cuda.is_available():
        raise ValueError("Qwen evaluation requires CUDA")
    output.mkdir(parents=True)
    records = _read_jsonl(dataset / "validation.stopping.jsonl")
    record_index = {record["id"]: record for record in records}
    pairs = _read_jsonl(dataset / "validation.pairs.jsonl")
    processor_source = adapter if adapter and (adapter / "preprocessor_config.json").exists() else model_path
    processor = AutoProcessor.from_pretrained(
        processor_source, local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    rows: list[dict] = []
    started = time.perf_counter()
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record, images in load_examples(
                    dataset, "validation", "stopping", require_formal_training=True):
                if image_mode == "masked":
                    images = [Image.new("RGB", image.size, (127, 127, 127)) for image in images]
                _, inputs = tokenize_prompt(processor, record, images)
                inputs = inputs.to(model.device)
                start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False)
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                text = processor.batch_decode(
                    generated, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0]
                decision = outcome = error = None
                try:
                    decision, outcome = _parse(text)
                except ValueError as exc:
                    error = str(exc)
                row = {
                    "id": record["id"], "episode_id": record["episode_id"],
                    "asset_id": record["data_identity"]["asset_id"],
                    "decision_index": record["decision_index"],
                    "termination_kind": record.get("termination_kind"),
                    "stop_reason": record["stop_reason"],
                    "expected_decision": record["stop_target"],
                    "expected_outcome": record["stop_outcome"],
                    "predicted_decision": decision, "predicted_outcome": outcome,
                    "image_mode": image_mode, "error": error, "raw": text,
                    "input_tokens": int(inputs["input_ids"].shape[1]),
                    "output_tokens": int(generated.shape[1]),
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                }
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                del inputs, output_ids, generated
        per_step = {
            str(step): summarize_stopping(
                [row for row in rows if row["decision_index"] == step])
            for step in sorted({row["decision_index"] for row in rows})
        }
        report = {
            "version": 1, "status": "complete", "scope": "v5_visual_stopping",
            "model": str(model_path), "adapter": str(adapter) if adapter else None,
            "dataset": str(dataset), "image_mode": image_mode,
            "mask_protocol": (
                "Every image is replaced by a same-size RGB(127,127,127) image; text, image count, "
                "image positions, context, action history, stage, and budget are unchanged."
                if image_mode == "masked" else None),
            "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
            "validation_stopping_sha256": _sha256(dataset / "validation.stopping.jsonl"),
            "validation_pairs_sha256": _sha256(dataset / "validation.pairs.jsonl"),
            "decode": "greedy", "max_new_tokens": max_new_tokens,
            "overall": summarize_stopping(rows),
            "per_step": per_step,
            "strict_pairs": summarize_pairs(rows, pairs, record_index),
            "elapsed_seconds": time.perf_counter() - started,
            "frozen_test_used": False,
            "limitations": [
                "Labels are deterministic silhouette evidence, not independent human defect diagnosis.",
                "All-image masking is an intervention on pixels but is out of distribution for the model.",
                "The strict pair audit tests model-visible non-image equality; record IDs and image paths differ outside messages.",
            ],
        }
        return report
    except Exception as exc:
        report = {"version": 1, "status": "incomplete",
                  "error_type": type(exc).__name__, "error": str(exc)}
        raise
    finally:
        (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--image-mode", choices=("original", "masked"), default="original")
    args = parser.parse_args()
    report = evaluate(
        args.dataset.resolve(), args.model.resolve(), args.out.resolve(),
        adapter=args.adapter.resolve() if args.adapter else None,
        max_new_tokens=args.max_new_tokens, image_mode=args.image_mode)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
