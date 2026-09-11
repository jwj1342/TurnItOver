# Source this file; never execute it. The only place that loads modules.
#   source scripts/setup_env.sh
module purge
module load StdEnv/2023 python/3.12 nodejs/24.15.0

export TIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PLAYWRIGHT_BROWSERS_PATH="$TIO_ROOT/.cache/ms-playwright"
export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$TIO_ROOT${PYTHONPATH:+:$PYTHONPATH}"   # the package is used in place; no install step
export TIO_DATA_DIR="${TIO_DATA_DIR:-$TIO_ROOT/data}"

# Login node: repo-local venv. Inside a Slurm job: venv in node-local $SLURM_TMPDIR.
if [ -n "${SLURM_TMPDIR:-}" ]; then
  export TIO_VENV="$SLURM_TMPDIR/venv"
else
  export TIO_VENV="$TIO_ROOT/.venv"
fi
if [ -f "$TIO_VENV/bin/activate" ]; then
  source "$TIO_VENV/bin/activate"
fi
