#!/usr/bin/env python3
"""Evaluate explicit continue/stop decisions on every held-out causal prefix."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.oracle.supervision import load_examples
from turnitover.oracle.training import tokenize_prompt
from turnitover.verifier.contracts import parse_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse(text: str) -> tuple[str | None, str | None]:
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


def _summarize(rows: list[dict]) -> dict:
    decision_confusion = Counter((row["expected_decision"], row["predicted_decision"] or "invalid")
                                 for row in rows)
    outcome_confusion = Counter((row["expected_outcome"] or "none",
                                 row["predicted_outcome"] or "none")
                                for row in rows if row["expected_decision"] == "stop")
    episodes = {}
    for row in rows:
        episodes.setdefault(row["episode_id"], []).append(row)
    timing = Counter()
    timing_eligible = 0
    for episode_rows in episodes.values():
        episode_rows.sort(key=lambda row: row["decision_index"])
        expected = next((row["decision_index"] for row in episode_rows
                         if row["expected_decision"] == "stop"), None)
        if expected is None:
            continue
        timing_eligible += 1
        predicted = next((row["decision_index"] for row in episode_rows
                          if row["predicted_decision"] == "stop"), None)
        timing["missing" if predicted is None else
               "early" if predicted < expected else
               "late" if predicted > expected else "exact"] += 1
    n = len(rows)
    stop_rows = [row for row in rows if row["expected_decision"] == "stop"]
    return {
        "n": n,
        "episodes": len(episodes),
        "valid_output_rate": sum(row["predicted_decision"] is not None for row in rows) / n,
        "decision_accuracy": sum(row["predicted_decision"] == row["expected_decision"] for row in rows) / n,
        "premature_stop": decision_confusion[("continue", "stop")],
        "missed_stop": sum(row["predicted_decision"] != "stop" for row in stop_rows),
        "decision_confusion": {f"{expected}->{predicted}": count
                               for (expected, predicted), count in sorted(decision_confusion.items())},
        "stop_outcome_confusion": {f"{expected}->{predicted}": count
                                   for (expected, predicted), count in sorted(outcome_confusion.items())},
        "timing_eligible_episodes": timing_eligible,
        "first_stop_timing": dict(timing),
        "budget_exhausted": {
            "n": sum(row["stop_reason"] == "budget_exhausted_uncertain" for row in rows),
            "correct_uncertain_stop": sum(
                row["stop_reason"] == "budget_exhausted_uncertain"
                and row["predicted_decision"] == "stop"
                and row["predicted_outcome"] == "uncertain" for row in rows),
        },
    }


def evaluate(dataset: Path, model_path: Path, output: Path, *, adapter: Path | None,
             max_new_tokens: int) -> dict:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    if not torch.cuda.is_available():
        raise ValueError("Qwen evaluation requires CUDA")
    output.mkdir(parents=True)
    examples = list(load_examples(dataset, "validation", "stopping", require_formal_training=True))
    processor_source = adapter if adapter and (adapter / "preprocessor_config.json").exists() else model_path
    processor = AutoProcessor.from_pretrained(processor_source, local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    rows = []
    started = time.perf_counter()
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record, images in examples:
                _, inputs = tokenize_prompt(processor, record, images)
                inputs = inputs.to(model.device)
                start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                text = processor.batch_decode(generated, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
                decision = outcome = error = None
                try:
                    decision, outcome = _parse(text)
                except ValueError as exc:
                    error = str(exc)
                payload = json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])
                row = {
                    "id": record["id"], "episode_id": record["episode_id"],
                    "asset_id": record["data_identity"]["asset_id"],
                    "decision_index": record["decision_index"], "remaining": payload["remaining"],
                    "history_length": len(payload["history"]), "image_count": len(images),
                    "evidence_available": bool(record["evidence"]["covered_label_indices"]),
                    "expected_decision": record["stop_target"],
                    "expected_outcome": record["stop_outcome"], "stop_reason": record["stop_reason"],
                    "predicted_decision": decision, "predicted_outcome": outcome,
                    "error": error, "raw": text, "input_tokens": int(inputs["input_ids"].shape[1]),
                    "output_tokens": int(generated.shape[1]),
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                }
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
        stage_evidence = {}
        for step in sorted({row["decision_index"] for row in rows}):
            for evidence in (False, True):
                subset = [row for row in rows if row["decision_index"] == step
                          and row["evidence_available"] == evidence]
                if subset:
                    stage_evidence[f"step={step},evidence={str(evidence).lower()}"] = _summarize(subset)
        report = {
            "version": 1, "status": "complete", "scope": "causal_prefix_stopping",
            "model": str(model_path), "adapter": str(adapter) if adapter else None,
            "dataset": str(dataset), "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
            "validation_stopping_sha256": _sha256(dataset / "validation.stopping.jsonl"),
            "max_new_tokens": max_new_tokens, "decode": "greedy",
            "overall": _summarize(rows), "same_stage_evidence": stage_evidence,
            "per_asset": {asset: _summarize([row for row in rows if row["asset_id"] == asset])
                          for asset in sorted({row["asset_id"] for row in rows})},
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": [
                "Prefixes are real and causal, but evidence-present and evidence-absent rows are not image-counterfactual pairs.",
                "The final exported prefix is the latest possible stop point; a missing stop cannot be classified as later within that episode.",
            ],
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
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    report = evaluate(args.dataset.resolve(), args.model.resolve(), args.out.resolve(),
                      adapter=args.adapter.resolve() if args.adapter else None,
                      max_new_tokens=args.max_new_tokens)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
