Remote root: `/home/laq/experiments/kimi-v4-increment-v1` on `laq@10.201.129.199`.
The isolated venv inherits the existing qwen environment and adds peft 0.18.0.
The specification is `env-spec.json`. Both comparison arms use this same environment.
Run the following command verbatim from the local training worktree. It validates seeded CUDA dispatch, every new row through the real v4 loader/processor and assistant mask, and real Qwen LoRA forward/backward without an optimizer step. Success requires both WITNESS and REAL_QWEN_FORWARD_BACKWARD and docs/remote-smoke.json status passed.

```bash
ssh -S /tmp/kimi-v4-ssh-%r@%h:%p laq@10.201.129.199 'cd /home/laq/experiments/kimi-v4-increment-v1/code && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false ../venv/bin/python ../docs/remote-smoke.py > ../docs/remote-smoke.log 2>&1'
```

No password is stored in files. SSH authentication uses the active session control socket.
