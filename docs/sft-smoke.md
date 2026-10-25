# Qwen SFT integration smoke

This smoke uses two procedural 32×32 PNG objects, one Train and one Validation
object, with no private or downloaded sample data. A seeded random Qwen3-VL has
one text layer, one vision layer, 64 text hidden dimensions and 19,533,792
parameters. Only the public Qwen tokenizer/processor is reused; no pretrained
model weights are loaded. The target is `uncertain`, never a hidden-label pass.

For the CPU smoke, the tested package versions are torch 2.6.0, torchvision
0.21.0, transformers 4.57.6, peft 0.18.0, accelerate 1.11.0 and
huggingface_hub 0.36.0. Install the ordinary requirements first, then these
packages in an isolated Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.6.0 torchvision==0.21.0
python -m pip install transformers==4.57.6 peft==0.18.0 accelerate==1.11.0 huggingface_hub==0.36.0
```

From the repository root in that environment, with a local Qwen3-VL
processor directory:

```bash
export QWEN_PROCESSOR=/path/to/Qwen3-VL-2B-Instruct
CUDA_VISIBLE_DEVICES='' USE_TF=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/smoke_qwen_sft.py --processor "$QWEN_PROCESSOR" \
  --out output/sft-pr-smoke --seed 20261025
```

Use a new output directory on each invocation. The processor directory needs
the tokenizer, image/video preprocessing configurations and chat template;
`model.safetensors` is not read. The existing download helper can populate this
directory, or it can be an already audited local model directory.

The script generates the fixture and random model, calls the **formal training
entry point** for one Evidence diagnosis optimizer step on CPU, saves the LoRA
checkpoint, checks that its four LoRA B tensors changed from zero, and calls
the **formal evaluator** twice. The second evaluation reloads the saved adapter.
Both perform real greedy two-token generation and must have identical input
and evaluation protocol hashes. This is deliberately too small to learn JSON.
Invalid outputs remain in the denominator.

The independent 2026-10-25 final-code rerun exited **0**, with loss **11.905364990234375**. Base and SFT
each evaluated one Validation case from one independent object, generated two
tokens, and produced invalid JSON (format validity **0/1** for each). This
demonstrates execution, checkpoint persistence/reload and error accounting;
it provides no evidence for diagnostic ability or Evidence SFT effectiveness.
This does not validate 2B/4B/8B GPU memory, throughput or numerical behavior.

The actual reused environment was Python 3.12.12, torch 2.6.0+cu124,
transformers 4.57.6, peft 0.18.0; all execution was on CPU with CUDA hidden.
This differs from the torch 2.7.1 pin in `requirements-qwen.txt`, so that pinned
environment is not certified by this run. The final-code rerun used a worktree-local venv inheriting the existing Torch runtime; missing dependencies were installed only in that venv. No shared environment was modified.
The run emitted upstream tokenizer regex, deprecated `torch_dtype`, and frozen
vision gradient-checkpoint warnings, plus automatic `pad_token_id` selection. The nonzero adapter assertion confirms
that the text LoRA optimizer update occurred despite the frozen vision warning.

[The compact run record](sft-smoke-run.json) includes raw generated text,
loss, split/protocol identities, code and artifact SHA-256 values. Full local
records are in `output/sft-pr-smoke-doccheck/{smoke.json,train,base,sft}`; generated
weights, image files and local logs are excluded from Git. The generated
`annotations.private.jsonl` contains only procedural empty labels; the filename
matches the audited loader interface and does not denote actual private data.
Runtime/timing-bearing files can differ between invocations. Seeds, source
code and the stable dataset/model/checkpoint hashes enable a focused comparison.
