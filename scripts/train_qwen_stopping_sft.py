#!/usr/bin/env python3
"""Train a stopping-only Qwen3-VL LoRA on visible causal prefixes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.oracle.supervision import load_examples
from turnitover.oracle.training import tokenize_example


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def train(dataset: Path, model_path: Path, output: Path, *, seed: int, epochs: int,
          learning_rate: float, lora_rank: int, lora_alpha: int) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    if not torch.cuda.is_available():
        raise ValueError("Qwen SFT requires CUDA")
    output.mkdir(parents=True)
    records = list(load_examples(dataset, "train", "stopping", require_formal_training=True))
    if not records:
        raise ValueError("No stopping examples")
    labels = Counter(record["stop_target"] for record, _ in records)
    if set(labels) != {"continue", "stop"}:
        raise ValueError(f"Both stopping labels are required: {labels}")
    weights = {label: len(records) / (len(labels) * count) for label, count in labels.items()}
    termination = Counter(str(record.get("termination_kind")) for record, _ in records)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    manifest = {
        "version": 1, "status": "running", "scope": "visual_prefix_stopping_only",
        "source_base_revision": _revision(),
        "trainer_sha256": _sha256(Path(__file__)),
        "seed": seed, "epochs": epochs,
        "learning_rate": learning_rate, "examples": len(records),
        "label_counts": dict(labels), "termination_kind_counts": dict(termination),
        "loss": {
            "type": "assistant-token mean cross entropy with inverse-frequency decision weights",
            "label_weights": weights,
        },
        "lora": {"rank": lora_rank, "alpha": lora_alpha, "dropout": 0.0,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
        "train_stopping_sha256": _sha256(dataset / "train.stopping.jsonl"),
        "base_model": str(model_path),
        "base_model_config_sha256": _sha256(model_path / "config.json"),
        "invocation": [sys.executable, *sys.argv],
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "frozen_test_used": False, "privileged_oracle_actions_used": False,
        "limitations": [
            "This adapter learns only continue/stop from current visible evidence and budget.",
            "Forced budget terminations are reported separately and retain fail or uncertain from visible evidence.",
            "Shape-consistency supervision does not establish general joint-motion diagnosis capability.",
        ],
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_path, loss_path = output / "train.log.jsonl", output / "loss.jsonl"
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=manifest["lora"]["target_modules"]))
        manifest["trainable_parameters"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        manifest["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate)
        order = list(range(len(records)))
        losses: list[float] = []
        raw_by_label: dict[str, list[float]] = {label: [] for label in labels}
        step = 0
        model.train()
        for epoch in range(epochs):
            random.Random(seed + epoch).shuffle(order)
            for position, index in enumerate(order):
                record, images = records[index]
                batch = tokenize_example(processor, record, images).to(model.device)
                optimizer.zero_grad(set_to_none=True)
                raw_loss = model(**batch).loss
                label = record["stop_target"]
                loss = raw_loss * weights[label]
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss at step {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                raw_value, value = float(raw_loss.detach().cpu()), float(loss.detach().cpu())
                row = {"step": step, "epoch": epoch, "position": position,
                       "id": record["id"], "label": label,
                       "termination_kind": record.get("termination_kind"),
                       "raw_loss": raw_value, "loss_weight": weights[label],
                       "weighted_loss": value, "learning_rate": learning_rate}
                with loss_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                if step % 10 == 0 or position + 1 == len(order):
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
                losses.append(value)
                raw_by_label[label].append(raw_value)
                step += 1
                del batch, raw_loss, loss
        checkpoint = output / "checkpoint"
        checkpoint.mkdir()
        model.save_pretrained(checkpoint, safe_serialization=True)
        processor.save_pretrained(checkpoint)
        manifest.update(
            status="complete", steps=step, checkpoint="checkpoint",
            weighted_loss={"first": losses[0], "last": losses[-1],
                           "minimum": min(losses), "mean": sum(losses) / len(losses)},
            raw_loss_by_label={label: {"first": values[0], "last": values[-1],
                                       "minimum": min(values), "mean": sum(values) / len(values)}
                               for label, values in raw_by_label.items()})
        return manifest
    except Exception as exc:
        manifest.update(status="incomplete", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()
    report = train(args.dataset.resolve(), args.model.resolve(), args.out.resolve(),
                   seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate,
                   lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
