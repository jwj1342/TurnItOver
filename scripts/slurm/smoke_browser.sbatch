#!/bin/bash
#SBATCH --account=def-zhouyang
#SBATCH --job-name=tio-smoke
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# Verifies that a compute node can run headless Chromium + the harness, then runs the unit
# and browser test suites. Submit from the repo root after scripts/bootstrap_login.sh.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/setup_env.sh
virtualenv --no-download "$TIO_VENV" >/dev/null
source "$TIO_VENV/bin/activate"
pip install --no-index -q -r requirements.txt

python scripts/check_browser.py
pytest
pytest -m browser
