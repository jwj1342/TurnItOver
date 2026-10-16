"""LoRA SFT for controlled matched-state motion diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import random

from turnitover.oracle.motion_diagnosis import load_motion_examples
from turnitover.oracle.training import tokenize_example


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError("Fresh output and CUDA are required")
    args.out.mkdir(parents=True); checkpoint = args.out / "checkpoint"
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    examples = [(record, images) for record, images, _ in load_motion_examples(args.dataset, "train", args.supervision)]
    manifest = {"status": "running", "task": "motion_diagnosis", "supervision": args.supervision,
        "examples": len(examples), "epochs": args.epochs, "steps_planned": len(examples) * args.epochs,
        "seed": args.seed, "learning_rate": args.learning_rate,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "dataset": str(args.dataset), "dataset_manifest_sha256": sha(args.dataset / "manifest.json"),
        "train_sha256": sha(args.dataset / "train.jsonl"), "base_model": str(args.model),
        "base_model_config_sha256": sha(args.model / "config.json"), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    (args.out / "run.json").write_text(json.dumps(manifest, indent=2))
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False; model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=manifest["lora"]["target_modules"]))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    losses=[]; step=0
    with (args.out / "loss.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            order=list(range(len(examples))); random.Random(args.seed + epoch).shuffle(order)
            for position, index in enumerate(order):
                record, images = examples[index]; batch = tokenize_example(processor, record, images).to(model.device)
                optimizer.zero_grad(set_to_none=True); loss=model(**batch).loss
                if not torch.isfinite(loss): raise ValueError(f"Non-finite loss at {step}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                value=float(loss.detach().cpu()); losses.append(value)
                row={"step":step,"epoch":epoch,"position":position,"id":record["id"],"loss":value}
                log.write(json.dumps(row)+"\n"); log.flush(); print(json.dumps(row),flush=True); step+=1
    checkpoint.mkdir(); model.save_pretrained(checkpoint, safe_serialization=True); processor.save_pretrained(checkpoint)
    manifest.update(status="complete", steps=step, checkpoint="checkpoint",
                    loss={"first":losses[0],"last":losses[-1],"minimum":min(losses),"mean":sum(losses)/len(losses)},
                    checkpoint_sha256=sha(checkpoint / "adapter_model.safetensors"))
    (args.out / "run.json").write_text(json.dumps(manifest, indent=2)); print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--supervision",choices=("ordinary","evidence"),required=True); p.add_argument("--seed",type=int,default=20261016)
    p.add_argument("--epochs",type=int,default=3); p.add_argument("--learning-rate",type=float,default=2e-4)
    p.add_argument("--lora-rank",type=int,default=8); p.add_argument("--lora-alpha",type=int,default=16)
    a=p.parse_args(); a.dataset=a.dataset.resolve(); a.model=a.model.resolve(); a.out=a.out.resolve(); main(a)
