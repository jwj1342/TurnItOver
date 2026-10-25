#!/usr/bin/env python3
"""CPU integration smoke: synthetic images, tiny random Qwen, real LoRA and generation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_dataset(root):
    """Two independent procedural objects; no downloaded or private sample data."""
    from PIL import Image, ImageDraw
    root.mkdir()
    annotations = []
    for index, split in enumerate(("train", "validation")):
        uid = f"synthetic-{split}"
        image = Image.new("RGB", (32, 32), "white")
        ImageDraw.Draw(image).rectangle((4 + index, 5, 23, 26), fill=(40, 80 + index * 40, 180))
        image.save(root / f"{uid}.png")
        digest = sha256(root / f"{uid}.png")
        identity = {"source_dataset": "sft-synthetic-smoke-v1", "source_asset_id": uid,
                    "asset_id": uid, "split_group": uid, "formal_training": True,
                    "source_identity_sha256": hashlib.sha256(f"sft-synthetic-smoke-v1:{uid}".encode()).hexdigest(),
                    "source_content_sha256": digest, "reference_sha": digest[:16],
                    "candidate_sha": digest[:16]}
        images = [f"{uid}.png", f"{uid}.png"]
        context = {"task": "synthetic geometry", "parts": ["panel"], "joints": [],
                   "views": ["front"], "runtime_properties": [], "action_types": ["request_view"],
                   "max_triangles": 100, "max_draw_calls": 10}
        payload = {"context": context, "remaining": 0, "image_order": images,
                   "history": [{"step": 0, "image_ref": images[1],
                                "action": {"type": "request_view", "view_id": "front"}}]}
        target = {"verdict": {"status": "uncertain", "confidence": 0.0, "findings": [],
                              "summary": "Insufficient evidence.", "limitations": ["Synthetic smoke only."]}}
        row = {"id": f"{uid}:verdict", "episode_id": uid, "terminal": True,
               "data_identity": identity, "images": images, "image_sha256": [digest, digest],
               "evidence": {"covered_label_indices": [], "uncovered_label_indices": []},
               "messages": [{"role": "user", "content": "Return one JSON verdict.\n" + json.dumps(payload)},
                            {"role": "assistant", "content": json.dumps(target)}]}
        (root / f"{split}.episodes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        annotations.append({"id": uid, "split": split, "reference_sha": identity["reference_sha"],
                            "candidate_sha": identity["candidate_sha"], "data_identity": identity, "labels": []})
    (root / "annotations.private.jsonl").write_text("".join(json.dumps(row) + "\n" for row in annotations), encoding="utf-8")
    write_json(root / "manifest.json", {"status": "complete", "formal_training": True,
               "dataset_version": "sft-synthetic-smoke-v1", "purpose": "integration smoke only",
               "groups": {"synthetic-train": "train", "synthetic-validation": "validation"}})


def make_model(processor_source, root, seed):
    import torch
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    torch.manual_seed(seed)
    processor = AutoProcessor.from_pretrained(processor_source, local_files_only=True, trust_remote_code=False)
    processor.image_processor.size = {"shortest_edge": 1024, "longest_edge": 1024}
    config = Qwen3VLConfig(
        text_config={"vocab_size": len(processor.tokenizer), "hidden_size": 64,
                     "intermediate_size": 128, "num_hidden_layers": 1, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 4096,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
                     "bos_token_id": processor.tokenizer.bos_token_id,
                     "eos_token_id": processor.tokenizer.eos_token_id},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 64, "num_position_embeddings": 16,
                       "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "deepstack_visual_indexes": []},
        tie_word_embeddings=True)
    model = Qwen3VLForConditionalGeneration(config)
    model.save_pretrained(root)
    processor.save_pretrained(root)
    return sum(p.numel() for p in model.parameters())


def run(processor_source, output, seed):
    import torch
    import transformers
    import peft
    from train_qwen_episode_sft import train
    from evaluate_qwen_episode_sft import evaluate
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    make_dataset(output / "dataset")
    parameters = make_model(processor_source, output / "model", seed)
    dataset = output / "dataset"
    manifest_sha = sha256(dataset / "manifest.json")
    training = train(dataset, output / "model", output / "train", seed=seed, epochs=1,
                     learning_rate=2e-4, lora_rank=2, lora_alpha=4, task="diagnosis",
                     supervision="evidence", max_steps=1, device="cpu", dataset_manifest_sha256=manifest_sha)
    from safetensors.torch import load_file
    adapter_state = load_file(output / "train/checkpoint/adapter_model.safetensors")
    updated = [name for name, value in adapter_state.items() if "lora_B" in name and torch.any(value != 0)]
    if not updated:
        raise AssertionError("LoRA B tensors remained zero; optimizer update was not demonstrated")
    reports = {}
    for name, adapter in (("base", None), ("sft", output / "train/checkpoint")):
        reports[name] = evaluate(dataset, output / "model", output / name, adapter=adapter,
                                 max_new_tokens=2, task="diagnosis", split="validation", seed=seed,
                                 device="cpu", dataset_manifest_sha256=manifest_sha)
        predictions = [json.loads(line) for line in (output / name / "predictions.jsonl").read_text().splitlines()]
        if len(predictions) != 1 or predictions[0]["output_tokens"] != 2:
            raise AssertionError("Expected one actual two-token inference output")
    assert training["status"] == "complete" and training["steps"] == 1
    assert reports["base"]["protocol"] == reports["sft"]["protocol"]
    artifacts = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    result = {"status": "complete", "scope": "integration smoke, not a research result",
              "seed": seed, "device": "cpu", "random_model_parameters": parameters,
              "python": platform.python_version(), "torch": torch.__version__,
              "transformers": transformers.__version__, "peft": peft.__version__,
              "training_steps": training["steps"], "training_loss": training["loss"],
              "nonzero_lora_B_tensors": len(updated),
              "independent_objects": {"train": 1, "validation": 1},
              "shared_protocol": reports["base"]["protocol"], "artifacts_sha256": artifacts}
    write_json(output / "smoke.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor", type=Path, required=True, help="Local Qwen3-VL tokenizer/processor directory; weights are not loaded")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    print(json.dumps(run(args.processor.resolve(), args.out.resolve(), args.seed), indent=2))


if __name__ == "__main__":
    main()
