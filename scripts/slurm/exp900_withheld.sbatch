#!/usr/bin/env bash
#SBATCH --job-name=tio-withheld
# Withheld-evidence evaluation: base model and every trained adapter of one size on the export with all exposing cells removed.
# Submit from the repo root; account, GPU, CPU, memory, time and --output on the command line (docs/vulcan-handoff.md).
# Required env: SIZE=2B|4B|8B. Optional: TIO_RUNS (default <repo>/tio-runs), TIO_MODELS (default <repo>/models).
set -uo pipefail
: "${SIZE:?SIZE=2B|4B|8B required}"
REPO=${TIO_ROOT:-$SLURM_SUBMIT_DIR}
export TIO_RUNS=${TIO_RUNS:-$REPO/tio-runs} TIO_MODELS=${TIO_MODELS:-$REPO/models}
cd "$REPO" && source scripts/setup_env.sh
virtualenv --no-download "$SLURM_TMPDIR/venv" > /dev/null || exit 1
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index -q -r scripts/exp900/requirements-900.txt || exit 1
export PYTHONPATH=$TIO_RUNS/code900 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
       CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p "$SLURM_TMPDIR/data" && cp -r "$TIO_RUNS/data900/three-tier-900-withheld-v1" "$SLURM_TMPDIR/data/" || exit 3
DATASET=$SLURM_TMPDIR/data/three-tier-900-withheld-v1/episode_export
mkdir -p "$SLURM_TMPDIR/models" && cp -r "$TIO_MODELS/Qwen3-VL-$SIZE-Instruct" "$SLURM_TMPDIR/models/" || exit 4
RUN=$TIO_RUNS/out/withheld-$SIZE-$SLURM_JOB_ID; mkdir -p "$RUN"
cp "$DATASET/PROVENANCE.json" "$DATASET/manifest.json" "$RUN/" 2>/dev/null
python -c "import torch,transformers,peft; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, peft.__version__)" > "$RUN/env.txt"
echo "repo $(git -C "$REPO" rev-parse HEAD)  code900 $(cat "$TIO_RUNS/code900/PROVENANCE.txt" 2>/dev/null)" >> "$RUN/env.txt"
bash "$REPO/scripts/exp900/run_withheld.sh" "$RUN" "$DATASET" "$SLURM_TMPDIR/models" "$SIZE"
