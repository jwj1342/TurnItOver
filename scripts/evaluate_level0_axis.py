"""Greedy pass/fail evaluation of base Qwen or a Level 0 LoRA adapter."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from turnitover.oracle.level0_axis import PROMPT, records_by_split, sha256
from turnitover.oracle.training import tokenize_prompt


def normalize(text: str) -> str:
    words = text.strip().lower().replace(".", " ").split()
    return words[0] if len(words) == 1 and words[0] in {"pass", "fail"} else "invalid"


def main(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if args.out.exists() or not torch.cuda.is_available(): raise ValueError("Fresh output and CUDA required")
    records = records_by_split(args.dataset, args.split, args.cohort)
    args.out.mkdir(parents=True)
    source = args.adapter if args.adapter else args.model
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if args.adapter: model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.eval(); counts = Counter(n=len(records)); rows = []
    with (args.out / "predictions.jsonl").open("w") as stream:
        for record in records:
            prompt_record = {"images": [str(record["image"])], "messages": [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": ""}]}
            _, inputs = tokenize_prompt(processor, prompt_record, [str(record["image"])])
            inputs = inputs.to(model.device); start = time.perf_counter()
            with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            ids = output[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            predicted = normalize(raw); correct = predicted == record["label"]
            row = {"case_id": record["case_id"], "expected": record["label"], "predicted": predicted, "correct": correct,
                   "raw": raw, "input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": int(ids.shape[1]),
                   "image_grid_thw": inputs["image_grid_thw"].tolist(), "pixel_values_shape": list(inputs["pixel_values"].shape), "seconds": time.perf_counter() - start}
            rows.append(row); stream.write(json.dumps(row) + "\n"); stream.flush(); print(json.dumps({k: v for k, v in row.items() if k != "raw"}), flush=True)
            counts["correct"] += int(correct); counts["invalid"] += int(predicted == "invalid")
    metrics = {"status": "complete", "cohort": args.cohort, "n": counts["n"], "correct": counts["correct"],
               "accuracy": counts["correct"] / counts["n"], "invalid": counts["invalid"], "prompt": PROMPT,
               "adapter": str(args.adapter) if args.adapter else None, "dataset_split_sha256": sha256(args.split),
               "cases": [{k: row[k] for k in ("case_id", "expected", "predicted", "correct")} for row in rows]}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True); parser.add_argument("--adapter", type=Path)
    parser.add_argument("--cohort", choices=["train", "heldout"], required=True); parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
