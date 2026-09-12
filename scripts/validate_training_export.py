"""Offline Qwen processor smoke: real image tensors and assistant-only token masks."""
import argparse
import json
from pathlib import Path

from transformers import AutoProcessor

from turnitover.oracle.supervision import load_examples
from turnitover.oracle.training import tokenize_example

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--model', type=Path, default=Path('models/Qwen3-VL-2B-Instruct'))
parser.add_argument('--kind', choices=('actions', 'diagnosis'), default='actions')
args = parser.parse_args()
if args.out.exists():
    raise ValueError('Output must not exist')
processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
rows = []
for split in ('train', 'validation', 'test'):
    for i, (record, images) in enumerate(load_examples(args.dataset, split, args.kind)):
        if i in (0, 5):
            batch = tokenize_example(processor, record, images)
            rows.append(dict(split=split, id=record['id'], images=len(images),
                             tokens=batch['input_ids'].shape[1],
                             supervised_tokens=int((batch['labels'] != -100).sum())))
        if i == 5:
            break
if len(rows) != 6:
    raise ValueError('Expected six validation examples across all splits')
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(dict(status='complete', model=str(args.model), examples=rows), indent=2))
print(args.out.read_text())
