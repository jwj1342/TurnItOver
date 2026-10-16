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
    parser.add_argument("--all-processor-rows", action="store_true",
                        help="Run the real processor and assistant-only mask check on every action/diagnosis row")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must not exist")
    processor = None
    if args.model is not None:
        from transformers import AutoProcessor
        from turnitover.oracle.training import tokenize_example
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    report = {"status": "complete", "dataset": str(args.dataset), "loaded": {}, "processor": [],
              "processor_summary": {"rows_checked": 0, "rows_failed": 0}}
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
            selected = rows if args.all_processor_rows else rows[:1] + rows[-1:]
            for record, images in selected:
                batch = tokenize_example(processor, record, images)
                labels = batch["labels"]
                supervised = (labels[0] != -100).nonzero()
                first_supervised = int(supervised[0])
                if first_supervised <= 0 or not bool((labels[0, :first_supervised] == -100).all()):
                    report["processor_summary"]["rows_failed"] += 1
                    raise ValueError(f"Assistant-only prefix mask failed: {record['id']}")
                report["processor_summary"]["rows_checked"] += 1
                report["processor"].append({
                    "kind": kind, "id": record["id"], "images": len(images),
                    "pixel_values_shape": list(batch["pixel_values"].shape),
                    "input_tokens": int(batch["input_ids"].shape[1]),
                    "supervised_tokens": int((labels != -100).sum()),
                    "first_supervised_token": first_supervised,
                    "assistant_only_prefix_mask": True})
    args.out.write_text(json.dumps(report, indent=2))
    print(args.out.read_text())
