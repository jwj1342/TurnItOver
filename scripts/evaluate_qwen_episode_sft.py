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
from turnitover.oracle.evaluation_examples import load_evaluation_examples
from turnitover.oracle.inference_prompt import tokenize_inference_prompt
from turnitover.taxonomy import load_taxonomy
from turnitover.oracle.evaluation_snapshot import create_snapshot
from turnitover.telemetry import git_state
from turnitover.oracle.sft_metrics import (
    PROTOCOL_VERSION, object_bootstrap, public_input_digest, score_prediction, summarize,
    summarize_by_difficulty,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_record(record, images, model_call, *, task="diagnosis",
                   inference_mode="observation-wise", withheld=()):
    """Share the observation inference contract with repair; score only afterward."""
    if task == "diagnosis" and inference_mode == "observation-wise":
        from turnitover.oracle.observation_inference import infer_observationwise
        result = infer_observationwise(record, images, model_call, withheld=withheld)
        row = score_prediction(record, json.dumps({"verdict": result["verdict"]}), task)
        row.update(observations=result["observations"],
                   findings_metadata=result["findings_metadata"], **result["usage"])
        row["model_calls"] = len(result["observations"])
        return row
    result = model_call(record, images)
    row = score_prediction(record, result["raw"], task)
    row.update({key: value for key, value in result.items() if key != "raw"})
    row["model_calls"] = 1
    return row


def evaluate(dataset: Path, model_path: Path, output: Path, *, adapter: Path | None,
             max_new_tokens: int, dataset_manifest_sha256: str, task="diagnosis",
             split="validation", seed=20261013, bootstrap_samples=0, device="cuda",
             training_seed=None, inference_mode="observation-wise", withheld=()) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, GenerationConfig

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
    if inference_mode not in ("observation-wise", "sweep"):
        raise ValueError("Unknown inference mode")
    if withheld and (task != "diagnosis" or inference_mode != "observation-wise"):
        raise ValueError("Withheld observations require observation-wise diagnosis")
    actual_manifest_sha = _sha256(dataset / "manifest.json")
    if actual_manifest_sha != dataset_manifest_sha256:
        raise ValueError("Dataset manifest SHA mismatch")
    kind = "episodes" if task == "diagnosis" else "stopping"
    annotation_sha = _sha256(dataset / "annotations.private.jsonl")
    split_sha = _sha256(dataset / f"{split}.{kind}.jsonl")
    examples = (list(load_evaluation_examples(dataset, split))
                if task == 'diagnosis' and inference_mode == 'observation-wise'
                else list(load_training_examples(dataset, split, task, "evidence")))
    if not examples:
        raise ValueError("No evaluation examples")
    records = [record for record, _ in examples]
    manifest = json.loads((dataset / "manifest.json").read_text())
    processor_files = sorted(path for path in model_path.iterdir() if path.is_file() and ("token" in path.name or "processor" in path.name or path.name == "chat_template.jinja"))
    processor_sha = hashlib.sha256(json.dumps({path.name: _sha256(path) for path in processor_files}, sort_keys=True).encode()).hexdigest()
    defect_ids = sorted(manifest.get("experiment_taxonomy", load_taxonomy().ids)) if task == "diagnosis" else []
    generation_config_path = model_path / "generation_config.json"
    if generation_config_path.exists():
        generation_config = GenerationConfig.from_pretrained(model_path, local_files_only=True)
    else:
        generation_config = GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False))
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = False
    generation_config.num_beams = 1
    model_type = json.loads((model_path / "config.json").read_text()).get("model_type")
    processor = None
    if model_type == "qwen3_5":
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model_eos = generation_config.eos_token_id
        eos_ids = list(model_eos) if isinstance(model_eos, (list, tuple)) else ([model_eos] if model_eos is not None else [])
        chat_eos = processor.tokenizer.eos_token_id
        if chat_eos is not None:
            eos_ids.append(chat_eos)
        generation_config.eos_token_id = sorted(set(eos_ids))
    effective_generation = {key: value for key, value in generation_config.to_dict().items()
                            if key not in ("_from_model_config", "_commit_hash", "transformers_version")}
    effective_generation_sha = hashlib.sha256(json.dumps(effective_generation, sort_keys=True).encode()).hexdigest()
    protocol = {"effective_generation_config_sha256": effective_generation_sha, "defect_ids": defect_ids, "processor_sha256": processor_sha, "device": device, "version": PROTOCOL_VERSION, "task": task, "split": split,
                "dataset_version": manifest.get("dataset_version", manifest.get("run_id", manifest.get("version"))),
                "dataset_manifest_sha256": actual_manifest_sha,
                "annotations_sha256": annotation_sha,
                "split_sha256": split_sha,
                "public_input_sha256": public_input_digest(records),
                "inference_mode": inference_mode if task == "diagnosis" else "stopping",
                "withheld_observations": list(withheld),
                "max_new_tokens": max_new_tokens, "do_sample": False, "seed": seed,
                "n": len(records), "independent_objects": len({row["data_identity"]["source_identity_sha256"] for row in records})}
    output.mkdir(parents=True)
    report = {"status": "incomplete", "protocol": protocol, "git": git_state(REPO_ROOT),
              "training_seed": training_seed}
    rows = []
    started = time.perf_counter()
    try:
        snapshot = create_snapshot(output, dataset, records, split, task, defect_ids)
        source_hashes = snapshot["manifest"]["source_sha256"]
        if (source_hashes["dataset_manifest"] != protocol["dataset_manifest_sha256"]
                or source_hashes["annotations"] != protocol["annotations_sha256"]
                or _sha256(dataset / f"{split}.{kind}.jsonl") != split_sha):
            raise ValueError("Dataset changed while creating scoring snapshot")
        report["scoring_snapshot_sha256"] = _sha256(snapshot["root"] / "manifest.json")
        report["adapter_files_sha256"] = {
            path.name: _sha256(path) for path in sorted(adapter.iterdir())
            if path.is_file() and (path.name.startswith("adapter_") or path.name == "run.json")
        } if adapter else None
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The base processor is mandatory for both base and adapters.
        if processor is None:
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa").to(device)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        model.eval()
        def model_call(record, images):
            _, inputs = tokenize_inference_prompt(processor, record, images, model_type=model_type)
            inputs = inputs.to(model.device)
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            return {"raw": processor.batch_decode(generated, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)[0],
                    "input_tokens": int(inputs["input_ids"].shape[1]),
                    "output_tokens": int(generated.shape[1]),
                    "elapsed_ms": (time.perf_counter() - start) * 1000}
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record, images in examples:
                row = predict_record(record, images, model_call, task=task,
                                     inference_mode=inference_mode, withheld=withheld)
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps({"id": row["id"], "valid": row["valid"], "error": row["error"]}), flush=True)
        metrics = summarize(rows, task, defect_ids)
        metrics["model_calls"] = sum(row["model_calls"] for row in rows)
        metrics["input_tokens"] = sum(row.get("input_tokens", 0) for row in rows)
        metrics["output_tokens"] = sum(row.get("output_tokens", 0) for row in rows)
        metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
        metrics["by_difficulty"] = summarize_by_difficulty(
            rows, task, defect_ids, bootstrap_samples=bootstrap_samples, seed=seed)
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
        if (output / "predictions.jsonl").is_file():
            report["predictions_sha256"] = _sha256(output / "predictions.jsonl")
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
    parser.add_argument("--training-seed", type=int, help="Seed of the evaluated training run, if known")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--inference-mode", choices=("observation-wise", "sweep"),
                        default="observation-wise")
    parser.add_argument("--withhold-observation", action="append", default=[],
                        help="Observation ID to omit before inference; repeat to omit several")
    args = parser.parse_args()
    report = evaluate(args.dataset.resolve(), args.model.resolve(), args.out.resolve(),
                      adapter=args.adapter.resolve() if args.adapter else None,
                      max_new_tokens=args.max_new_tokens, task=args.task, split=args.split,
                      seed=args.seed, bootstrap_samples=args.bootstrap_samples, device=args.device,
                      dataset_manifest_sha256=args.dataset_manifest_sha256, training_seed=args.training_seed,
                      inference_mode=args.inference_mode, withheld=args.withhold_observation)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
