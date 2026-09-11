#!/bin/bash
# One-time setup on a LOGIN node (needs internet for npm and the Playwright browser download).
# Everything it does is lightweight. Re-running is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh

if [ ! -f "$TIO_VENV/bin/activate" ]; then
  virtualenv --no-download "$TIO_VENV"
fi
source "$TIO_VENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python -m playwright install chromium-headless-shell

# npm ci needs a committed lock file; the first run generates it with npm install.
( cd web && if [ -f package-lock.json ]; then npm ci; else npm install; fi && npm run build )

echo "bootstrap done. Next: sbatch scripts/slurm/smoke_browser.sbatch"
