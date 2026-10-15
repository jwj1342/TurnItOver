#!/usr/bin/env python3
"""Continue a v4 evidence LoRA with balanced v4-diagnosis/v5-stopping replay."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.oracle.multitask_replay import (
    load_protocol_examples,
    replay_groups,
    stopping_class_weights,
)
from turnitover.oracle.sft_data import supervision_digest
from turnitover.oracle.training import tokenize_example


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def train(v4_dataset: Path, v5_dataset: Path, model_path: Path, initial_adapter: Path,
          output: Path, *, seed: int, epochs: int, learning_rate: float,
          stopping_per_diagnosis: int) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    if not torch.cuda.is_available():
        raise ValueError("Qwen SFT requires CUDA")
    output.mkdir(parents=True)
    diagnosis, stopping = load_protocol_examples(v4_dataset, v5_dataset)
    stop_weights = stopping_class_weights(stopping)
    groups = replay_groups(len(diagnosis), len(stopping), seed=seed,
                           stopping_per_diagnosis=stopping_per_diagnosis)
    diagnosis_exposures = Counter(group["diagnosis"] for group in groups)
    protocol_counts = Counter(
        record["input_protocol"] for record, _ in [*diagnosis, *stopping])
    manifest = {
        "version": 1, "status": "running", "scope": "v4_diagnosis_v5_stopping_replay",
        "source_base_revision": _revision(),
        "trainer_sha256": _sha256(Path(__file__)),
        "seed": seed, "epochs": epochs, "learning_rate": learning_rate,
        "base_model": str(model_path), "initial_adapter": str(initial_adapter),
        "initial_adapter_sha256": _sha256(initial_adapter / "adapter_model.safetensors"),
        "datasets": {
            "v4_diagnosis": str(v4_dataset), "v5_stopping": str(v5_dataset),
            "v4_manifest_sha256": _sha256(v4_dataset / "manifest.json"),
            "v5_manifest_sha256": _sha256(v5_dataset / "manifest.json"),
        },
        "mixing": {
            "protocol_counts": dict(protocol_counts),
            "diagnosis_unique_examples": len(diagnosis),
            "stopping_unique_examples": len(stopping),
            "stopping_per_diagnosis": stopping_per_diagnosis,
            "optimizer_groups_per_epoch": len(groups),
            "diagnosis_exposures_per_epoch": len(groups),
            "diagnosis_exposure_range": [min(diagnosis_exposures.values()),
                                         max(diagnosis_exposures.values())],
            "stopping_exposures_per_epoch": len(stopping),
            "objective_loss_per_group": "0.5 diagnosis + 0.5 mean class-balanced stopping",
            "stopping_class_weights": stop_weights,
            "selection": "all v5 stopping rows once; deterministic cycling over all v4 evidence terminal diagnoses",
        },
        "supervision_sha256": {
            "diagnosis": supervision_digest([record for record, _ in diagnosis]),
            "stopping": supervision_digest([record for record, _ in stopping]),
        },
        "input_protocols_are_separate": True,
        "privileged_oracle_actions_used": False,
        "frozen_test_used": False,
        "invocation": [sys.executable, *sys.argv],
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "limitations": [
            "v4 diagnosis uses one reference and v5 stopping uses eight aligned state references.",
            "No action target is trained; browser closed-loop evaluation uses a fixed public schedule.",
            "v5 shape-consistency stopping does not establish joint-axis diagnosis capability.",
        ],
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    loss_path = output / "loss.jsonl"
    try:
        processor_source = initial_adapter if (initial_adapter / "preprocessor_config.json").exists() else model_path
        processor = AutoProcessor.from_pretrained(
            processor_source, local_files_only=True, trust_remote_code=False)
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        model = PeftModel.from_pretrained(
            base, initial_adapter, local_files_only=True, is_trainable=True)
        manifest["trainable_parameters"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        manifest["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate)
        losses = {"diagnosis": [], "stopping": [], "combined": []}
        optimizer_step = 0
        model.train()
        with loss_path.open("w", encoding="utf-8") as stream:
            for epoch in range(epochs):
                epoch_groups = replay_groups(
                    len(diagnosis), len(stopping), seed=seed + epoch,
                    stopping_per_diagnosis=stopping_per_diagnosis)
                for position, group in enumerate(epoch_groups):
                    optimizer.zero_grad(set_to_none=True)
                    record, images = diagnosis[group["diagnosis"]]
                    batch = tokenize_example(processor, record, images).to(model.device)
                    diagnosis_loss = model(**batch).loss
                    if not torch.isfinite(diagnosis_loss):
                        raise ValueError(f"Non-finite diagnosis loss at {optimizer_step}")
                    (diagnosis_loss * .5).backward()
                    diagnosis_value = float(diagnosis_loss.detach().cpu())
                    del batch, diagnosis_loss
                    stop_values = []
                    stop_rows = group["stopping"]
                    for index in stop_rows:
                        stop_record, stop_images = stopping[index]
                        batch = tokenize_example(processor, stop_record, stop_images).to(model.device)
                        raw_loss = model(**batch).loss
                        weighted = raw_loss * stop_weights[stop_record["stop_target"]]
                        if not torch.isfinite(weighted):
                            raise ValueError(f"Non-finite stopping loss at {optimizer_step}")
                        (weighted * (.5 / len(stop_rows))).backward()
                        stop_values.append(float(weighted.detach().cpu()))
                        del batch, raw_loss, weighted
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    stopping_value = sum(stop_values) / len(stop_values)
                    combined = .5 * (diagnosis_value + stopping_value)
                    row = {
                        "optimizer_step": optimizer_step, "epoch": epoch, "position": position,
                        "diagnosis_id": record["id"],
                        "stopping_ids": [stopping[index][0]["id"] for index in stop_rows],
                        "diagnosis_loss": diagnosis_value,
                        "class_balanced_stopping_loss": stopping_value,
                        "combined_loss": combined, "learning_rate": learning_rate,
                    }
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    if optimizer_step % 10 == 0 or position + 1 == len(epoch_groups):
                        print(json.dumps(row), flush=True)
                    losses["diagnosis"].append(diagnosis_value)
                    losses["stopping"].append(stopping_value)
                    losses["combined"].append(combined)
                    optimizer_step += 1
        checkpoint = output / "checkpoint"
        checkpoint.mkdir()
        model.save_pretrained(checkpoint, safe_serialization=True)
        processor.save_pretrained(checkpoint)
        manifest.update(
            status="complete", optimizer_steps=optimizer_step,
            micro_examples=epochs * (len(stopping) + len(groups)), checkpoint="checkpoint",
            loss={key: {"first": values[0], "last": values[-1],
                        "minimum": min(values), "mean": sum(values) / len(values)}
                  for key, values in losses.items()})
        return manifest
    except Exception as exc:
        manifest.update(status="incomplete", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-dataset", type=Path, required=True)
    parser.add_argument("--v5-dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--stopping-per-diagnosis", type=int, default=4)
    args = parser.parse_args()
    report = train(
        args.v4_dataset.resolve(), args.v5_dataset.resolve(), args.model.resolve(),
        args.initial_adapter.resolve(), args.out.resolve(), seed=args.seed,
        epochs=args.epochs, learning_rate=args.learning_rate,
        stopping_per_diagnosis=args.stopping_per_diagnosis)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
