#!/usr/bin/env python3
"""Evaluate base or LoRA Qwen on identical held-out terminal Episode inputs."""
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
from turnitover.verifier.contracts import Context, parse_decision, parse_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(verdict) -> set[tuple[str, str | None, tuple[str, ...]]]:
    return {(row.defect_id, row.joint_id, tuple(sorted(row.parts))) for row in verdict.findings}


def _finding_evidence(verdict):
    return {(row.defect_id, row.joint_id, tuple(sorted(row.parts))): set(row.evidence_steps)
            for row in verdict.findings}


def evaluate(dataset: Path, model_path: Path, output: Path, *, adapter: Path | None,
             max_new_tokens: int) -> dict:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    if not torch.cuda.is_available():
        raise ValueError("Qwen evaluation requires CUDA")
    output.mkdir(parents=True)
    examples = list(load_examples(dataset, "validation", "episodes",
                                  require_formal_training=True, terminal_only=True))
    processor_source = adapter if adapter and (adapter / "preprocessor_config.json").exists() else model_path
    processor = AutoProcessor.from_pretrained(processor_source, local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    counts = Counter(n=len(examples))
    per_reason = {}
    per_asset = {}
    started = time.perf_counter()
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record, images in examples:
                target_value = json.loads(record["messages"][1]["content"])
                payload = json.loads(record["messages"][0]["content"].rsplit("\n", 1)[-1])
                context = Context(**payload["context"])
                observed = {row["step"] for row in payload["history"]}
                target = parse_decision(target_value, context, observed, final_only=True)
                _, inputs = tokenize_prompt(processor, record, images)
                inputs = inputs.to(model.device)
                start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                text = processor.batch_decode(generated, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
                prediction = None
                error = None
                try:
                    prediction = parse_decision(parse_json(text), context, observed, final_only=True)
                except ValueError as exc:
                    error = str(exc)
                    counts["invalid"] += 1
                expected_signature = _signature(target)
                predicted_signature = _signature(prediction) if prediction else set()
                expected_evidence = _finding_evidence(target)
                predicted_evidence = _finding_evidence(prediction) if prediction else {}
                unsupported = sum(
                    signature not in expected_evidence
                    or not steps <= expected_evidence.get(signature, set())
                    for signature, steps in predicted_evidence.items())
                counts["status_correct"] += int(prediction is not None and prediction.status == target.status)
                if prediction is not None:
                    counts[f"predicted_status:{prediction.status}"] += 1
                counts["exact_verdict"] += int(prediction is not None and prediction.status == target.status
                                                and predicted_signature == expected_signature)
                counts["stop_correct"] += int(prediction is not None)
                counts["unsupported_findings"] += unsupported
                counts["predicted_findings"] += len(predicted_signature)
                evidence_consistent = (prediction is not None and unsupported == 0
                                       and not (prediction.status == "pass" and target.status != "pass"))
                counts["evidence_consistent"] += int(evidence_consistent)
                counts["false_fail"] += int(prediction is not None and prediction.status == "fail"
                                             and target.status != "fail")
                counts["false_pass"] += int(prediction is not None and prediction.status == "pass"
                                             and target.status != "pass")
                for label in expected_signature | predicted_signature:
                    counts["tp" if label in expected_signature & predicted_signature else
                           "fp" if label in predicted_signature else "fn"] += 1
                reason = record["termination_reason"]
                bucket = per_reason.setdefault(reason, Counter())
                bucket["n"] += 1
                bucket["valid"] += int(prediction is not None)
                bucket["status_correct"] += int(prediction is not None and prediction.status == target.status)
                asset = record["data_identity"]["asset_id"]
                asset_bucket = per_asset.setdefault(asset, Counter())
                asset_bucket["n"] += 1
                asset_bucket["valid"] += int(prediction is not None)
                asset_bucket["status_correct"] += int(
                    prediction is not None and prediction.status == target.status)
                asset_bucket["exact_verdict"] += int(
                    prediction is not None and prediction.status == target.status
                    and predicted_signature == expected_signature)
                asset_bucket["false_fail"] += int(
                    prediction is not None and prediction.status == "fail" and target.status != "fail")
                asset_bucket["unsupported_findings"] += unsupported
                row = {"id": record["id"], "episode_id": record["episode_id"],
                       "data_identity": record["data_identity"], "termination_reason": reason,
                       "expected_status": target.status,
                       "expected_findings": [list(item) for item in sorted(expected_signature)],
                       "predicted_status": prediction.status if prediction else None,
                       "predicted_findings": [list(item) for item in sorted(predicted_signature)],
                       "unsupported_findings": unsupported,
                       "evidence_consistent": evidence_consistent,
                       "error": error, "raw": text,
                       "input_tokens": int(inputs["input_ids"].shape[1]),
                       "output_tokens": int(generated.shape[1]),
                       "elapsed_ms": (time.perf_counter() - start) * 1000}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps({key: value for key, value in row.items() if key != "raw"}), flush=True)
        tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
        report = {
            "version": 1, "status": "complete", "scope": "fixed_observation_terminal_diagnosis_and_stop",
            "model": str(model_path), "adapter": str(adapter) if adapter else None,
            "dataset": str(dataset), "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
            "validation_episodes_sha256": _sha256(dataset / "validation.episodes.jsonl"),
            "max_new_tokens": max_new_tokens, "n": counts["n"],
            "valid_output_rate": counts["stop_correct"] / counts["n"],
            "final_verdict_rate": counts["stop_correct"] / counts["n"],
            "stop_accuracy": counts["stop_correct"] / counts["n"],
            "status_accuracy": counts["status_correct"] / counts["n"],
            "exact_verdict_accuracy": counts["exact_verdict"] / counts["n"],
            "finding_micro": {"tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None},
            "false_fail_rate": counts["false_fail"] / counts["n"],
            "false_pass_rate": counts["false_pass"] / counts["n"],
            "evidence_consistency": {
                "consistent_output_rate": counts["evidence_consistent"] / counts["n"],
                "unsupported_findings": counts["unsupported_findings"],
                "predicted_findings": counts["predicted_findings"],
            },
            "expected_statuses": dict(Counter(json.loads(record["messages"][1]["content"])["verdict"]["status"]
                                              for record, _ in examples)),
            "predicted_statuses": {status: counts[f"predicted_status:{status}"]
                                   for status in ("pass", "fail", "uncertain")},
            "per_termination_reason": {key: dict(value) for key, value in per_reason.items()},
            "per_asset": {key: dict(value) for key, value in sorted(per_asset.items())},
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": [
                "Validation contains seven held-out PartNet-Mobility assets.",
                "Inputs use frozen teacher-acquired observations; active action selection is not evaluated.",
                "No pass-qualified example exists, so pass behavior cannot be measured.",
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
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    report = evaluate(args.dataset.resolve(), args.model.resolve(), args.out.resolve(),
                      adapter=args.adapter.resolve() if args.adapter else None,
                      max_new_tokens=args.max_new_tokens)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
