"""Action-only LoRA: one optimizer update per complete empirical target group."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from turnitover.oracle.supervision import load_action_groups
from turnitover.oracle.training import tokenize_example, weighted_assistant_loss
from turnitover.verifier.contracts import Context, parse_decision


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # Validate every Train group and its image files before allocating GPU/output.
    groups = list(load_action_groups(args.dataset, 'train'))
    for group in groups:
        for record, _ in group:
            payload = json.loads(record['messages'][0]['content'].rsplit('\n', 1)[1])
            target = json.loads(record['messages'][1]['content'])
            if set(target) != {'action'} or payload['remaining'] <= 0:
                raise ValueError('Only next-action targets with positive budget are allowed')
            parse_decision(target, Context(**payload['context']), {o['step'] for o in payload['history']})
    if not groups or args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Nonempty complete Train data, fresh output and CUDA required')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.out.mkdir(parents=True)
    def digest(p):
        with p.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    provenance = {str(p.relative_to(args.dataset)): digest(p) for p in [
        args.dataset/'manifest.json', args.dataset/'train.actions.jsonl']}
    for group in groups:
        for record, _ in group:
            for name in record['images']:
                provenance[name] = digest(args.dataset/name)
    run = dict(status='running', task='action_only_empirical_teacher_bc',
               seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate,
               lora=dict(rank=4, alpha=8, dropout=0., target_modules=['q_proj','k_proj','v_proj','o_proj']),
               groups=len(groups), variants=sum(map(len, groups)),
               objective='equal group mean of probability-weighted assistant-token mean cross entropy',
               optimizer='AdamW default betas/eps/weight_decay; constant LR; grad clip 1',
               dataset=str(args.dataset.resolve()), data_sha256=provenance,
               base_model=str(args.model.resolve()), base_config_sha256=digest(args.model/'config.json'),
               code_sha256={str(p): digest(p) for p in [Path(__file__),
                   Path('turnitover/oracle/training.py'), Path('turnitover/oracle/supervision.py')]},
               torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    path = args.out/'run.json'
    path.write_text(json.dumps(run, indent=2))
    run['base_weights_sha256'] = {p.name: digest(p) for p in sorted(args.model.glob('*.safetensors'))}
    start = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0., bias='none',
                          task_type='CAUSAL_LM', target_modules=run['lora']['target_modules']))
        model.train()
        run['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
        steps = 0
        for epoch in range(args.epochs):
            order = list(range(len(groups)))
            random.Random(args.seed+epoch).shuffle(order)
            for index in order:
                optimizer.zero_grad(set_to_none=True)
                variants = []
                # Sequential backwards accumulate exactly the packed weighted group
                # gradient without keeping all multimodal variants in GPU memory.
                for record, images in groups[index]:
                    batch = tokenize_example(processor, record, images).to(model.device)
                    output = model(**batch)
                    raw = weighted_assistant_loss(output.logits, batch['labels'], [1.])
                    weight = record['loss_weight']
                    if not torch.isfinite(raw):
                        raise ValueError('Nonfinite assistant loss')
                    (raw*weight).backward()
                    variants.append(dict(id=record['id'], weight=weight, loss=float(raw.detach().cpu())))
                    del batch, raw, output
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                row = dict(step=steps, epoch=epoch, group=index, variants=variants,
                           loss=sum(v['loss']*v['weight'] for v in variants))
                with (args.out/'loss.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
                steps += 1
        model.save_pretrained(args.out/'checkpoint', safe_serialization=True)
        processor.save_pretrained(args.out/'checkpoint')
        run.update(status='complete', steps=steps,
                   checkpoint_sha256={p.name: digest(p) for p in (args.out/'checkpoint').iterdir() if p.is_file()},
                   peak_cuda_allocated=torch.cuda.max_memory_allocated())
    except Exception as exc:
        run.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        run['elapsed_seconds'] = time.perf_counter()-start
        path.write_text(json.dumps(run, indent=2))
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error('Positive epochs and learning rate required')
    train(args)
