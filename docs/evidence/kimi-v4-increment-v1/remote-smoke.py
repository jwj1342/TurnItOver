"""Verbatim remote environment witness, no optimizer step or checkpoint writes."""
import hashlib
import json
from pathlib import Path
import sys

root = Path('/home/laq/experiments/kimi-v4-increment-v1')
sys.path.insert(0, str(root / 'code'))
import torch
import peft
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from turnitover.oracle.sft_data import load_multitask_training_examples, supervision_digest
from turnitover.oracle.training import tokenize_example, tokenize_prompt

torch.manual_seed(20261013)
x = torch.randn(8, 8, device='cuda')
assert torch.isfinite(x @ x).all()
print('WITNESS', (x @ x).shape, torch.cuda.get_device_name(), flush=True)
processor = AutoProcessor.from_pretrained(root / 'models/Qwen3-VL-2B-Instruct', local_files_only=True)
examples = list(load_multitask_training_examples(root / 'data/experiment', 'train', 'evidence'))
new = [(r, images) for r, images in examples if r['condition'] == 'visible_only_interactive_teacher']
assert len(new) == 10
summaries = []
for record, images in new:
    batch = tokenize_example(processor, record, images)
    _, prefix = tokenize_prompt(processor, record, images)
    n = prefix['input_ids'].shape[1]
    assert torch.all(batch['labels'][:, :n] == -100)
    assert torch.equal(batch['labels'][:, n:], batch['input_ids'][:, n:])
    summaries.append({'id': record['id'], 'prefix_tokens': n, 'assistant_tokens': int((batch['labels'] != -100).sum())})
model = Qwen3VLForConditionalGeneration.from_pretrained(root / 'models/Qwen3-VL-2B-Instruct', local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., bias='none', task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj']))
model.train()
batch = tokenize_example(processor, *new[0]).to(model.device)
loss = model(**batch).loss
assert torch.isfinite(loss)
loss.backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
model.zero_grad(set_to_none=True)
model.eval()
_, generation_inputs = tokenize_prompt(processor, *new[0])
with torch.inference_mode():
    generated = model.generate(**generation_inputs.to(model.device), max_new_tokens=2, do_sample=False)
assert generated.shape[1] > generation_inputs['input_ids'].shape[1]
result = {'generation_witness_tokens': int(generated.shape[1] - generation_inputs['input_ids'].shape[1]), 'status': 'passed', 'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__, 'gpu': torch.cuda.get_device_name(), 'loss': float(loss.detach()), 'new_processor_rows': summaries, 'supervision_sha256': supervision_digest([r for r, _ in examples])}
(root / 'docs/remote-smoke.json').write_text(json.dumps(result, indent=2))
print('REAL_QWEN_FORWARD_BACKWARD', json.dumps(result), flush=True)
