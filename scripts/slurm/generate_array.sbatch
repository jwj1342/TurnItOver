#!/bin/bash
#SBATCH --account=def-zhouyang
#SBATCH --job-name=tio-generate
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Data-generation array job: each task owns one shard, fully determined by (config, task id).
#   CONFIG=configs/generate_toy.yaml sbatch scripts/slurm/generate_array.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/setup_env.sh
virtualenv --no-download "$TIO_VENV" >/dev/null
source "$TIO_VENV/bin/activate"
pip install --no-index -q -r requirements.txt

CONFIG="${CONFIG:-configs/generate_toy.yaml}"
python -m turnitover --log-json generate --config "$CONFIG" \
  --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
