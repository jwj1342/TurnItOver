#!/bin/bash
#SBATCH --account=def-zhouyang
#SBATCH --job-name=tio-qwen
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/setup_env.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$TIO_ROOT/models/.hf-cache"
export OMP_NUM_THREADS=4
exec "$TIO_ROOT/.venv-qwen/bin/python" -m turnitover verify \
  --local-model "$TIO_ROOT/models/Qwen3-VL-2B-Instruct" "$@"
