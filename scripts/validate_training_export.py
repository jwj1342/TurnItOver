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
parser.add_argument('--kind', choices=('actions', 'diagnosis', 'episodes'), default='actions')
args = parser.parse_args()
if args.out.exists():
    raise ValueError('Output must not exist')
processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
rows = []
seen_splits = set()
for split in ('train', 'validation', 'test'):
    for i, (record, images) in enumerate(load_examples(
            args.dataset, split, args.kind, require_formal_training=args.kind == 'episodes',
            terminal_only=args.kind == 'episodes')):
        if i < 2:
            batch = tokenize_example(processor, record, images)
            labels = batch['labels']
            first_supervised = int((labels[0] != -100).nonzero()[0])
            rows.append(dict(split=split, id=record['id'], images=len(images),
                             image_tensor_shape=list(batch['pixel_values'].shape),
                             tokens=batch['input_ids'].shape[1],
                             first_supervised_token=first_supervised,
                             supervised_tokens=int((labels != -100).sum())))
            seen_splits.add(split)
        if i == 1:
            break
if not {'train', 'validation'} <= seen_splits:
    raise ValueError('Expected real train and validation examples')
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(dict(status='complete', model=str(args.model), examples=rows), indent=2))
print(args.out.read_text())
