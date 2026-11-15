#!/bin/bash
#SBATCH --job-name=tio-vllm
# OpenAI-compatible vLLM server used as the repair generator, one replica per job. Submit from the repo root with account,
# GPU, CPU, memory, time and --output on the command line, e.g.
#   MODEL=models/Qwen3.8-27B-FP8 sbatch --account=$ACCT --gpus-per-node=l40s:1 --cpus-per-task=8 --mem=64G \
#     --time=12:00:00 --output=tio-runs/logs/%x-%j.out scripts/slurm/vllm_serve.sbatch
# Writes http://<node>:<port>/v1 to $TIO_RUNS/vllm/<jobid>.endpoint. Requests need the key in ~/.tio-vllm-token
# (COMPATIBLE_API_KEY); repair jobs pass GEN_NO_PROXY=<node> so they bypass the Squid http_proxy set by the prolog.
# Required env: MODEL. Optional: SERVED_NAME (default basename of MODEL), TP (1), MAX_MODEL_LEN (65536), MAX_NUM_SEQS (32;
#   Qwen3.5-family hybrid attention needs one Mamba cache block per sequence, about 185 fit on one L40S),
#   VLLM_EXTRA (extra `vllm serve` arguments), TIO_RUNS (default <repo>/tio-runs).
set -euo pipefail
: "${MODEL:?MODEL=<weights dir> required}"
REPO=${TIO_ROOT:-$SLURM_SUBMIT_DIR}
TIO_RUNS=${TIO_RUNS:-$REPO/tio-runs}
cd "$REPO" && TIO_ENV=vllm source scripts/setup_env.sh
virtualenv --no-download "$SLURM_TMPDIR/venv" > /dev/null
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --no-deps -q -r scripts/vllm/requirements-vllm.txt     # full lock; see the file header
export HF_HUB_OFFLINE=1 HF_HOME=$SLURM_TMPDIR/hf XDG_CACHE_HOME=$SLURM_TMPDIR/cache VLLM_CACHE_ROOT=$SLURM_TMPDIR/vllm-cache \
       OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
VLLM_API_KEY=$(cat ~/.tio-vllm-token)
export VLLM_API_KEY
PORT=$((20000 + SLURM_JOB_ID % 20000))
mkdir -p "$TIO_RUNS/vllm"
echo "http://$(hostname):$PORT/v1" > "$TIO_RUNS/vllm/$SLURM_JOB_ID.endpoint"
python -c "import torch, vllm, transformers; print(torch.__version__, vllm.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
echo "model $MODEL revision $(cat "$MODEL/REVISION.txt" 2>/dev/null)  repo $(git rev-parse HEAD)"
exec vllm serve "$MODEL" --served-model-name "${SERVED_NAME:-$(basename "$MODEL")}" --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "${TP:-1}" --max-model-len "${MAX_MODEL_LEN:-65536}" --gpu-memory-utilization 0.92 \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 16}' ${VLLM_EXTRA:-}
