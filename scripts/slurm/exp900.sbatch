#!/usr/bin/env bash
#SBATCH --job-name=tio-900
# Three-tier-900 diagnosis jobs (her training and evaluation scripts in $TIO_RUNS/code900, unchanged).
# Submit from the repo root. Account, GPU, CPU, memory, time and --output are given on the sbatch command line
# (see docs/vulcan-handoff.md). Required env: MODE=sft|base|eval.
#   MODE=sft  SIZE=8B SEEDS=20261213 SUPS=ordinary   train and evaluate
#   MODE=base SIZES="2B 8B"                          zero-shot and pixel baselines
#   MODE=eval SIZE=2B SEEDS=... SUPS=...             evaluate existing checkpoints in $TIO_RUNS/out
# Optional env: SPLIT=validation|test, TIO_RUNS (default <repo>/tio-runs), TIO_MODELS (default <repo>/models).
set -uo pipefail
: "${MODE:?MODE=sft|base|eval required}"
REPO=${TIO_ROOT:-$SLURM_SUBMIT_DIR}
export TIO_RUNS=${TIO_RUNS:-$REPO/tio-runs} TIO_MODELS=${TIO_MODELS:-$REPO/models}
if [ "$MODE" = base ]; then SIZES=${SIZES:-"2B 4B 8B"}; else SIZES=${SIZE:?SIZE required for MODE=$MODE}; fi
cd "$REPO" && source scripts/setup_env.sh
virtualenv --no-download "$SLURM_TMPDIR/venv" > /dev/null || exit 1
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index -q -r scripts/exp900/requirements-900.txt || exit 1
export PYTHONPATH=$TIO_RUNS/code900 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
       CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# dataset to node-local disk (6600 small files), verified against the release checksum
ZIP=$TIO_RUNS/data900/dl/900-private-candidates.zip
echo "185d2d6ba7ca2b6d2fb7426d407e50c9682a8632e3a827f18502cab9c309e2e0  $ZIP" | sha256sum -c - || exit 3
mkdir -p "$SLURM_TMPDIR/data" && (cd "$SLURM_TMPDIR/data" && unzip -q "$ZIP") || exit 3
DATASET=$SLURM_TMPDIR/data/three-tier-900-sft-v1/episode_export
mkdir -p "$SLURM_TMPDIR/models"
for size in $SIZES; do cp -r "$TIO_MODELS/Qwen3-VL-$size-Instruct" "$SLURM_TMPDIR/models/" || exit 4; done
RUN=$TIO_RUNS/out/level900-${MODE}-${SIZES// /}-$SLURM_JOB_ID
mkdir -p "$RUN"
python -c "import torch,transformers,peft; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, peft.__version__)" > "$RUN/env.txt"
echo "repo $(git -C "$REPO" rev-parse HEAD)  code900 $(cat "$TIO_RUNS/code900/PROVENANCE.txt" 2>/dev/null)" >> "$RUN/env.txt"
cd "$TIO_RUNS/code900"
export SIZES
bash "$REPO/scripts/exp900/run_900.sh" "$RUN" "$DATASET" "$SLURM_TMPDIR/models"
