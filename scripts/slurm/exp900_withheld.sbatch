#!/usr/bin/env bash
#SBATCH --job-name=tio-withheld
# Withheld-evidence evaluation: base model and every trained adapter of one size on the export with all exposing cells removed.
# Submit from the repo root; account, GPU, CPU, memory, time and --output on the command line (docs/vulcan-handoff.md).
# Required env: SIZE=2B|4B|8B. Optional: DERIVED (export under $TIO_RUNS/data900, default three-tier-900-withheld-v1),
#   MANIFEST_SHA (expected sha256 of its manifest.json; default is the withheld export's), TAG (run name prefix, default withheld),
#   CODE (her code dir, default $TIO_RUNS/code900; v1.1: $TIO_RUNS/code900-v11), CKPT_TAG (adapters from
#   $TIO_RUNS/out/$CKPT_TAG-sft-*, default level900; v1.1: level900v11),
#   TIO_RUNS (default <repo>/tio-runs), TIO_MODELS (default <repo>/models).
set -uo pipefail
: "${SIZE:?SIZE=2B|4B|8B required}"
REPO=${TIO_ROOT:-$SLURM_SUBMIT_DIR}
export TIO_RUNS=${TIO_RUNS:-$REPO/tio-runs} TIO_MODELS=${TIO_MODELS:-$REPO/models}
cd "$REPO" && source scripts/setup_env.sh
virtualenv --no-download "$SLURM_TMPDIR/venv" > /dev/null || exit 1
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index -q -r scripts/exp900/requirements-900.txt || exit 1
export CODE=${CODE:-$TIO_RUNS/code900} CKPT_TAG=${CKPT_TAG:-level900}
export PYTHONPATH=$CODE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
       CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
DERIVED=${DERIVED:-three-tier-900-withheld-v1}
export MANIFEST_SHA=${MANIFEST_SHA:-4931a1cb12ccff9a25cac321b1c59ec2f5206812d0dc4b466381c7623cf2ab24}
mkdir -p "$SLURM_TMPDIR/data" && cp -r "$TIO_RUNS/data900/$DERIVED" "$SLURM_TMPDIR/data/" || exit 3
DATASET=$SLURM_TMPDIR/data/$DERIVED/episode_export
mkdir -p "$SLURM_TMPDIR/models" && cp -r "$TIO_MODELS/Qwen3-VL-$SIZE-Instruct" "$SLURM_TMPDIR/models/" || exit 4
RUN=$TIO_RUNS/out/${TAG:-withheld}-$SIZE-$SLURM_JOB_ID; mkdir -p "$RUN"
cp "$DATASET/PROVENANCE.json" "$DATASET/manifest.json" "$RUN/" 2>/dev/null
python -c "import torch,transformers,peft; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, peft.__version__)" > "$RUN/env.txt"
echo "repo $(git -C "$REPO" rev-parse HEAD)  code900 $(cat "$CODE/PROVENANCE.txt" 2>/dev/null)  adapters $CKPT_TAG" >> "$RUN/env.txt"
bash "$REPO/scripts/exp900/run_withheld.sh" "$RUN" "$DATASET" "$SLURM_TMPDIR/models" "$SIZE"
