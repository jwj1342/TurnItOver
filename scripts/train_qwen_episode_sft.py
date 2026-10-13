#!/usr/bin/env python3
"""Minimal LoRA SFT for evidence-qualified terminal Episode decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.oracle.supervision import load_examples
from turnitover.oracle.training import tokenize_example


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_log(path: Path, row: dict) -> None:
    line = json.dumps(row, sort_keys=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


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
    log_path = output / "train.log.jsonl"
    loss_path = output / "loss.jsonl"
    checkpoint = output / "checkpoint"
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    examples = list(load_examples(dataset, "train", "episodes",
                                  require_formal_training=True, terminal_only=True))
    if not examples:
        raise ValueError("No formal terminal training examples")
    manifest = {
        "version": 1, "status": "running", "scope": "terminal_diagnosis_and_stop_only",
        "seed": seed, "epochs": epochs, "learning_rate": learning_rate,
        "lora": {"rank": lora_rank, "alpha": lora_alpha, "dropout": 0.0,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "dataset": str(dataset), "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
        "train_episodes_sha256": _sha256(dataset / "train.episodes.jsonl"),
        "base_model": str(model_path), "base_model_config_sha256": _sha256(model_path / "config.json"),
        "examples": len(examples), "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "limitations": [
            "Only terminal structured diagnosis/stop decisions are supervised.",
            "Privileged Oracle action targets are excluded.",
            "This small run validates the training path; it does not establish generalization.",
        ],
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_log(log_path, {"event": "load", "examples": len(examples), "gpu": manifest["gpu"]})
    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=manifest["lora"]["target_modules"]))
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        manifest.update(trainable_parameters=trainable, total_parameters=total)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate)
        order = list(range(len(examples)))
        global_step = 0
        losses = []
        model.train()
        for epoch in range(epochs):
            random.Random(seed + epoch).shuffle(order)
            for position, index in enumerate(order):
                record, images = examples[index]
                batch = tokenize_example(processor, record, images).to(model.device)
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss at step {global_step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                value = float(loss.detach().cpu())
                row = {"step": global_step, "epoch": epoch, "position": position,
                       "id": record["id"], "loss": value, "learning_rate": learning_rate}
                with loss_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                _write_log(log_path, {"event": "step", **row})
                losses.append(value)
                global_step += 1
                del batch, loss
        checkpoint.mkdir()
        model.save_pretrained(checkpoint, safe_serialization=True)
        processor.save_pretrained(checkpoint)
        manifest.update(status="complete", steps=global_step,
                        loss={"first": losses[0], "last": losses[-1],
                              "minimum": min(losses), "mean": sum(losses) / len(losses)},
                        checkpoint="checkpoint")
        _write_log(log_path, {"event": "complete", "steps": global_step, "loss": manifest["loss"]})
        return manifest
    except Exception as exc:
        manifest.update(status="incomplete", error_type=type(exc).__name__, error=str(exc))
        _write_log(log_path, {"event": "error", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--epochs", type=int, default=3)
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
