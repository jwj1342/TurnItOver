#!/usr/bin/env python3
"""Train one paired-observation verifier LoRA adapter through PyTRIO.

This CLI makes training requests only when explicitly executed, never at import.
PyTRIO exposes LoRA rank but not alpha; alpha=16 cannot be guaranteed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(args):
    # Validate before importing the SDK or creating a service client.
    if args.alpha is not None:
        raise ValueError("PyTRIO create_lora_training_client does not expose alpha; remove --alpha. Alpha 16 is not guaranteed.")
    if args.epochs < 1 or args.rank < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("epochs, rank and learning-rate must be positive and finite")
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}")
    manifest_path = args.dataset / "manifest.json"
    manifest_sha = sha256(manifest_path)
    if manifest_sha != args.dataset_manifest_sha256:
        raise ValueError("Dataset manifest SHA mismatch")

    from turnitover.oracle.paired_observations import is_paired
    from turnitover.oracle.sft_data import load_training_examples, supervision_digest
    from turnitover.oracle.sft_metrics import public_input_digest

    examples = list(load_training_examples(args.dataset, "train", "diagnosis", args.supervision))
    if not examples or any(not is_paired(record) for record, _ in examples):
        raise ValueError("Training requires nonempty paired-observation examples")
    records = [record for record, _ in examples]
    targets = [{"id": row["id"], "assistant": row["messages"][1]} for row in records]
    target_sha = hashlib.sha256(json.dumps(targets, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {
        "status": "running", "task": "diagnosis", "split": "train",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset_version": dataset_manifest.get("dataset_version"),
        "dataset_manifest_sha256": manifest_sha,
        "train_export_sha256": sha256(args.dataset / "train.episodes.jsonl"),
        "annotations_sha256": sha256(args.dataset / "annotations.private.jsonl"),
        "public_input_sha256": public_input_digest(records),
        "target_sha256": target_sha, "supervision_sha256": supervision_digest(records),
        "examples": len(examples), "steps": 0,
        "lora": {"rank": args.rank, "alpha": None, "alpha_status": "unavailable_in_pytrio_public_api"},
        "loss": "assistant_token_cross_entropy", "learning_rate": args.learning_rate,
    }
    args.out.mkdir(parents=True)
    run_path = args.out / "run.json"

    def save_report():
        run_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save_report()
    try:
        import pytrio
        from transformers import AutoImageProcessor
        from turnitover.oracle.pytrio_adapter import model_input

        client = pytrio.ServiceClient()
        training = client.create_lora_training_client(base_model=args.model, rank=args.rank, seed=args.seed)
        tokenizer = client.create_sampling_client(base_model=args.model).get_tokenizer()
        processor = AutoImageProcessor.from_pretrained(
            getattr(tokenizer, "name_or_path", args.model), use_fast=False)
        losses = []
        with (args.out / "loss.jsonl").open("w", encoding="utf-8") as log:
            for epoch in range(args.epochs):
                order = list(range(len(examples)))
                random.Random(args.seed + epoch).shuffle(order)
                for index in order:
                    record, images = examples[index]
                    datum = model_input(record, images, tokenizer, processor,
                                        completion=record["messages"][1]["content"], sdk=pytrio)
                    expected_tokens = int(datum.loss_fn_inputs["weights"].to_numpy().sum())
                    result = training.forward_backward([datum], loss_fn="cross_entropy").result()
                    if expected_tokens <= 0 or result.metrics.get("token_count") != expected_tokens:
                        raise ValueError("Service loss token count differs from assistant mask")
                    loss = float(result.metrics["loss_mean"])
                    if not math.isfinite(loss):
                        raise ValueError("Service returned non-finite loss")
                    training.optim_step(pytrio.AdamParams(learning_rate=args.learning_rate)).result()
                    row = {"step": report["steps"], "epoch": epoch, "id": record["id"],
                           "loss": loss, "target_tokens": expected_tokens,
                           "learning_rate": args.learning_rate}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    losses.append(loss)
                    report["steps"] += 1
                    save_report()
        saved = training.save_weights_for_sampler(name=f"paired-sft-{args.supervision}").result()
        report.update(status="complete", weights_path=str(saved.path),
                      loss_summary={"first": losses[0], "last": losses[-1], "mean": sum(losses) / len(losses)})
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save_report()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supervision", choices=("ordinary", "evidence"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, help="Unsupported by PyTRIO; specifying this option fails explicitly")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.out = args.out.resolve()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
