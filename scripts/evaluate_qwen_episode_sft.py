#!/usr/bin/env python3
"""Evaluate base and LoRA Qwen with one fixed-observation inference contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.oracle.sft_data import load_training_examples
from turnitover.oracle.training import tokenize_prompt
from turnitover.taxonomy import load_taxonomy
from turnitover.oracle.sft_metrics import (
    PROTOCOL_VERSION, object_bootstrap, public_input_digest, score_prediction, summarize,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(dataset: Path, model_path: Path, output: Path, *, adapter: Path | None,
             max_new_tokens: int, dataset_manifest_sha256: str, task="diagnosis",
             split="validation", seed=20261013, bootstrap_samples=0, device="cuda") -> dict:
    import torch
    from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    if max_new_tokens <= 0 or bootstrap_samples < 0:
        raise ValueError("Invalid generation or bootstrap budget")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Qwen evaluation requires CUDA")
    if task not in ("diagnosis", "stopping"):
        raise ValueError("Unknown task")
    actual_manifest_sha = _sha256(dataset / "manifest.json")
    if actual_manifest_sha != dataset_manifest_sha256:
        raise ValueError("Dataset manifest SHA mismatch")
    kind = "episodes" if task == "diagnosis" else "stopping"
    examples = list(load_training_examples(dataset, split, task, "evidence"))
    if not examples:
        raise ValueError("No evaluation examples")
    records = [record for record, _ in examples]
    manifest = json.loads((dataset / "manifest.json").read_text())
    processor_files = sorted(path for path in model_path.iterdir() if path.is_file() and ("token" in path.name or "processor" in path.name or path.name == "chat_template.jinja"))
    processor_sha = hashlib.sha256(json.dumps({path.name: _sha256(path) for path in processor_files}, sort_keys=True).encode()).hexdigest()
    defect_ids = sorted(load_taxonomy().ids) if task == "diagnosis" else []
    generation_config_path = model_path / "generation_config.json"
    if generation_config_path.exists():
        generation_config = GenerationConfig.from_pretrained(model_path, local_files_only=True)
    else:
        generation_config = GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False))
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = False
    effective_generation = {key: value for key, value in generation_config.to_dict().items()
                            if key not in ("_from_model_config", "_commit_hash", "transformers_version")}
    effective_generation_sha = hashlib.sha256(json.dumps(effective_generation, sort_keys=True).encode()).hexdigest()
    protocol = {"effective_generation_config_sha256": effective_generation_sha, "defect_ids": defect_ids, "processor_sha256": processor_sha, "device": device, "version": PROTOCOL_VERSION, "task": task, "split": split,
                "dataset_version": manifest.get("dataset_version", manifest.get("run_id", manifest.get("version"))),
                "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
                "annotations_sha256": _sha256(dataset / "annotations.private.jsonl"),
                "split_sha256": _sha256(dataset / f"{split}.{kind}.jsonl"),
                "public_input_sha256": public_input_digest(records),
                "max_new_tokens": max_new_tokens, "do_sample": False, "seed": seed,
                "n": len(records), "independent_objects": len({row["data_identity"]["source_identity_sha256"] for row in records})}
    output.mkdir(parents=True)
    report = {"status": "incomplete", "protocol": protocol}
    rows = []
    started = time.perf_counter()
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The base processor is mandatory for both base and adapters.
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa").to(device)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        model.eval()
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record, images in examples:
                _, inputs = tokenize_prompt(processor, record, images)
                inputs = inputs.to(model.device)
                start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, generation_config=generation_config)
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                raw = processor.batch_decode(generated, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)[0]
                row = score_prediction(record, raw, task)
                row.update(input_tokens=int(inputs["input_ids"].shape[1]), output_tokens=int(generated.shape[1]),
                           elapsed_ms=(time.perf_counter() - start) * 1000)
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps({"id": row["id"], "valid": row["valid"], "error": row["error"]}), flush=True)
        metrics = summarize(rows, task, defect_ids)
        report.update(version=3, status="complete", scope=f"fixed_observation_{task}",
                      model=str(model_path), adapter=str(adapter) if adapter else None,
                      model_config_sha256=_sha256(model_path / "config.json"),
                      generation_config_file_sha256=_sha256(generation_config_path) if generation_config_path.exists() else "absent",
                      effective_generation_config=effective_generation,
                      protocol=protocol, **metrics,
                      elapsed_seconds=time.perf_counter() - started,
                      limitations=["Frozen observations do not measure active evidence acquisition.",
                                   "Evidence support means agreement with audited target citations, not independent human grounding.",
                                   "Clean false positives require an explicit audited is_clean label; otherwise unavailable.",
                                   "Single supplied joint IDs are excluded from open joint localization."])
        if bootstrap_samples:
            report["bootstrap"] = object_bootstrap(rows, "status_accuracy" if task == "diagnosis" else "decision_accuracy",
                                                    samples=bootstrap_samples, seed=seed, task=task, defect_ids=defect_ids)
    except Exception as exc:
        report.update(status="incomplete", completed_cases=len(rows), error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=("diagnosis", "stopping"), default="diagnosis")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()
    report = evaluate(args.dataset.resolve(), args.model.resolve(), args.out.resolve(),
                      adapter=args.adapter.resolve() if args.adapter else None,
                      max_new_tokens=args.max_new_tokens, task=args.task, split=args.split,
                      seed=args.seed, bootstrap_samples=args.bootstrap_samples, device=args.device,
                      dataset_manifest_sha256=args.dataset_manifest_sha256)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
