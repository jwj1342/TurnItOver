#!/bin/bash
#SBATCH --account=def-zhouyang
#SBATCH --job-name=tio-cmd
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Run one command inside the project environment on a compute node:
#   sbatch scripts/slurm/run_cmd.sbatch python -m turnitover detectability --config configs/detectability_toy.yaml --out data/runs/detect_toy.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/setup_env.sh
virtualenv --no-download "$TIO_VENV" >/dev/null
source "$TIO_VENV/bin/activate"
pip install --no-index -q -r requirements.txt
exec "$@"
