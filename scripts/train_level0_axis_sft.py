"""One small Qwen3-VL LoRA run for the fixed Level 0 binary image task."""
from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path

from turnitover.oracle.level0_axis import PROMPT, records_by_split, sha256
from turnitover.oracle.training import tokenize_example


def record_for_sft(record: dict) -> dict:
    return {"images": [str(record["image"])], "messages": [
        {"role": "user", "content": PROMPT}, {"role": "assistant", "content": record["label"]}]}


def main(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError("Fresh output and CUDA are required")
    records = records_by_split(args.dataset, args.split, "train")
    args.out.mkdir(parents=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    first = tokenize_example(processor, record_for_sft(records[0]), [str(records[0]["image"])])
    preprocess = {k: list(v.shape) for k, v in first.items() if hasattr(v, "shape")}
    manifest = {"status": "running", "task": "level0_axis_binary", "epochs": args.epochs,
        "steps_planned": len(records) * args.epochs, "seed": args.seed, "learning_rate": args.learning_rate,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "dataset": str(args.dataset), "split_sha256": sha256(args.split), "base_model": str(args.model),
        "base_model_config_sha256": sha256(args.model / "config.json"), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "visual_preprocess": {"original_image": [542, 542], "position_order": "row-major: ref-rest, ref-limit, candidate-rest, candidate-limit", "processor_batch_shapes": preprocess}}
    (args.out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False; model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=manifest["lora"]["target_modules"]))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    losses = []; step = 0
    with (args.out / "loss.jsonl").open("w") as stream:
        for epoch in range(args.epochs):
            order = list(range(len(records))); random.Random(args.seed + epoch).shuffle(order)
            for position, index in enumerate(order):
                record = records[index]
                batch = tokenize_example(processor, record_for_sft(record), [str(record["image"])]).to(model.device)
                optimizer.zero_grad(set_to_none=True); loss = model(**batch).loss
                if not torch.isfinite(loss): raise ValueError(f"Non-finite loss at {step}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                value = float(loss.detach().cpu()); losses.append(value)
                row = {"step": step, "epoch": epoch, "position": position, "case_id": record["case_id"], "loss": value}
                stream.write(json.dumps(row) + "\n"); stream.flush(); print(json.dumps(row), flush=True); step += 1
    checkpoint = args.out / "checkpoint"; checkpoint.mkdir(); model.save_pretrained(checkpoint, safe_serialization=True); processor.save_pretrained(checkpoint)
    manifest.update(status="complete", steps=step, checkpoint="checkpoint", checkpoint_sha256=sha256(checkpoint / "adapter_model.safetensors"),
                    loss={"first": losses[0], "last": losses[-1], "minimum": min(losses), "mean": sum(losses) / len(losses)})
    (args.out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017); parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--lora-rank", type=int, default=8); parser.add_argument("--lora-alpha", type=int, default=16)
    main(parser.parse_args())
