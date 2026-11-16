#!/usr/bin/env bash
#SBATCH --job-name=tio-900
# Three-tier-900 diagnosis jobs (her training and evaluation scripts in $TIO_RUNS/code900, unchanged).
# Submit from the repo root. Account, GPU, CPU, memory, time and --output are given on the sbatch command line
# (see docs/vulcan-handoff.md). Required env: MODE=sft|base|eval.
#   MODE=sft  SIZE=8B SEEDS=20261213 SUPS=ordinary   train and evaluate
#   MODE=base SIZES="2B 8B"                          zero-shot and pixel baselines
#   MODE=eval SIZE=2B SEEDS=... SUPS=...             evaluate existing checkpoints in $TIO_RUNS/out
# Optional env: SPLIT=validation|test, TIO_RUNS (default <repo>/tio-runs), TIO_MODELS (default <repo>/models),
#   DATA=v1|v11 (default v1; v11 = three-tier-900-sft-v1.1-pre.1, clean cases target pass and 16 validation cases are
#   quarantined, with her code at 89e07db in $TIO_RUNS/code900-v11; runs are named level900v11-*).
set -uo pipefail
: "${MODE:?MODE=sft|base|eval required}"
REPO=${TIO_ROOT:-$SLURM_SUBMIT_DIR}
export TIO_RUNS=${TIO_RUNS:-$REPO/tio-runs} TIO_MODELS=${TIO_MODELS:-$REPO/models}
if [ "$MODE" = base ]; then SIZES=${SIZES:-"2B 4B 8B"}; else SIZES=${SIZE:?SIZE required for MODE=$MODE}; fi
cd "$REPO" && source scripts/setup_env.sh
virtualenv --no-download "$SLURM_TMPDIR/venv" > /dev/null || exit 1
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index -q -r scripts/exp900/requirements-900.txt || exit 1
if [ "${DATA:-v1}" = v11 ]; then
  ZIP=$TIO_RUNS/data900/v11-pre1-download/three-tier-900-sft-v1.1.zip
  ZIPSHA=b0816822e1cfc4039bd4414819a043740bab89a250237f61d3e10a1a86f29aec SUB=three-tier-900-sft-v1.1
  CODE=$TIO_RUNS/code900-v11 TAG=level900v11
  export MANIFEST_SHA=b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b
else
  ZIP=$TIO_RUNS/data900/dl/900-private-candidates.zip
  ZIPSHA=185d2d6ba7ca2b6d2fb7426d407e50c9682a8632e3a827f18502cab9c309e2e0 SUB=three-tier-900-sft-v1/episode_export
  CODE=$TIO_RUNS/code900 TAG=level900
  export MANIFEST_SHA=6d0fc5344810396e9454e83c58572f9a9ecdcae4cb53a477f59e9dd541fc6c25
fi
export RUN_TAG=$TAG
export PYTHONPATH=$CODE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
       CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# dataset to node-local disk (6600 small files), verified against the release checksum
echo "$ZIPSHA  $ZIP" | sha256sum -c - || exit 3
mkdir -p "$SLURM_TMPDIR/data" && (cd "$SLURM_TMPDIR/data" && unzip -q "$ZIP") || exit 3
DATASET=$SLURM_TMPDIR/data/$SUB
mkdir -p "$SLURM_TMPDIR/models"
for size in $SIZES; do cp -r "$TIO_MODELS/Qwen3-VL-$size-Instruct" "$SLURM_TMPDIR/models/" || exit 4; done
RUN=$TIO_RUNS/out/$TAG-${MODE}-${SIZES// /}-$SLURM_JOB_ID
mkdir -p "$RUN"
python -c "import torch,transformers,peft; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, peft.__version__)" > "$RUN/env.txt"
echo "repo $(git -C "$REPO" rev-parse HEAD)  code900 $(cat "$CODE/PROVENANCE.txt" 2>/dev/null)  data $SUB" >> "$RUN/env.txt"
cd "$CODE"
export SIZES
bash "$REPO/scripts/exp900/run_900.sh" "$RUN" "$DATASET" "$SLURM_TMPDIR/models"
