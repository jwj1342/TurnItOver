"""Load an interactive-teacher export with the formal SFT loader and, optionally, the Qwen processor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from turnitover.oracle.supervision import load_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Local Qwen processor for real image tensors and masks")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must not exist")
    processor = None
    if args.model is not None:
        from transformers import AutoProcessor
        from turnitover.oracle.training import tokenize_example
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    report = {"status": "complete", "dataset": str(args.dataset), "loaded": {}, "processor": []}
    for split in ("train", "validation"):
        for kind in ("actions", "diagnosis", "episodes"):
            rows = list(load_examples(args.dataset, split, kind, require_formal_training=True))
            report["loaded"][f"{split}.{kind}"] = {
                "rows": len(rows), "terminal": sum(bool(r["terminal"]) for r, _ in rows),
                "images": sum(len(images) for _, images in rows),
                "statuses": sorted({json.loads(r["messages"][1]["content"])["verdict"]["status"]
                                    for r, _ in rows if r["terminal"]})}
            if processor is None or split != "train" or kind == "episodes" or not rows:
                continue
            for record, images in rows[:1] + rows[-1:]:
                batch = tokenize_example(processor, record, images)
                labels = batch["labels"]
                report["processor"].append({
                    "kind": kind, "id": record["id"], "images": len(images),
                    "pixel_values_shape": list(batch["pixel_values"].shape),
                    "input_tokens": int(batch["input_ids"].shape[1]),
                    "supervised_tokens": int((labels != -100).sum()),
                    "first_supervised_token": int((labels[0] != -100).nonzero()[0])})
    args.out.write_text(json.dumps(report, indent=2))
    print(args.out.read_text())
