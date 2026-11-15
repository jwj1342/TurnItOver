#!/usr/bin/env bash
#SBATCH --job-name=tio-probe
# One short compute-node probe, run once after scripts/exp900/vulcan_preflight.sh: network paths to the generator
# (direct, cluster proxy, login-node relay), node resources, modules, wheelhouse venv, headless Chromium + harness,
# and the GPU when one is requested. Submit from the repo root, e.g.
#   sbatch --account=$ACCT --time=00:20:00 --cpus-per-task=2 --mem=8G --output=tio-runs/logs/%x-%j.out scripts/slurm/vulcan_probe.sbatch
# Optional env: GEN_PROXY (login-node relay printed by scripts/exp900/connect_proxy.py).
set -uo pipefail
REPO=${TIO_ROOT:-$SLURM_SUBMIT_DIR}
cd "$REPO"
section() { printf '\n== %s\n' "$*"; }
mask() { sed -E 's#//[^:/@]+:[^@]+@#//***:***@#g'; }
host_of() { grep "^$1=" .env 2>/dev/null | sed -E 's#^[^=]*=https?://([^/:]+).*#\1#'; }

section node
echo "host $(hostname)  job $SLURM_JOB_ID  partition ${SLURM_JOB_PARTITION:-?}  cpus ${SLURM_CPUS_PER_TASK:-?}  mem ${SLURM_MEM_PER_NODE:-?}M"
lscpu | grep -E 'Model name|^CPU\(s\)' ; df -h "$SLURM_TMPDIR" | tail -1
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU visible"

section proxy environment on the compute node
env | grep -iE '^(https?|no|all)_proxy=' | mask || echo "no proxy variables"

section generator endpoint
PROVIDER=$(grep '^TIO_GENERATOR_PROVIDER=' .env | cut -d= -f2)
case $PROVIDER in anthropic) VAR=ANTHROPIC_BASE_URL;; openai) VAR=OPENAI_BASE_URL;; openai_compatible) VAR=COMPATIBLE_BASE_URL;; gemini) VAR=GEMINI_BASE_URL;; *) VAR=;; esac
GEN_HOST=$(host_of "$VAR")
echo "provider $PROVIDER  model $(grep '^TIO_GENERATOR_MODEL=' .env | cut -d= -f2)  host ${GEN_HOST:+<from $VAR>}"
probe() { printf '%-34s' "$1"; shift; curl -sS -o /dev/null -m 15 -w 'HTTP %{http_code}  %{time_total}s\n' "$@" 2>&1 | tail -1 | mask; }
if [ -n "$GEN_HOST" ]; then
  probe "generator via cluster default" "https://$GEN_HOST/"
  probe "generator with proxies unset" --noproxy '*' "https://$GEN_HOST/"
  [ -n "${GEN_PROXY:-}" ] && probe "generator via GEN_PROXY relay" -x "$GEN_PROXY" "https://$GEN_HOST/"
fi
probe "huggingface.co" https://huggingface.co/
probe "github.com" https://github.com/
probe "pypi.org" https://pypi.org/simple/
echo "(HTTP 000 = not reachable; any other code means the network path works)"

section modules and wheelhouse venv
source scripts/setup_env.sh && echo "modules ok: $(python --version 2>&1), node $(node --version 2>&1)"
t0=$(date +%s)
virtualenv --no-download "$SLURM_TMPDIR/venv" > /dev/null && source "$SLURM_TMPDIR/venv/bin/activate" \
  && pip install --no-index -q -r requirements.txt && echo "venv from wheelhouse ok in $(( $(date +%s) - t0 ))s"

section headless Chromium and harness
python scripts/check_browser.py && echo "browser ok" || echo "BROWSER FAILED (run scripts/bootstrap_login.sh on the login node first)"

section done
echo "PROBE DONE $(date -Is)"
