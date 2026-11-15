#!/usr/bin/env bash
# Login-node preflight for the three-tier-900 experiments on Vulcan (light: queries and file checks only).
# Checks the git state, the transferred data against scripts/exp900/transfer-manifest.tsv, .env, modules, wheels,
# Slurm accounts, partitions, GPU types and per-user limits, and the login node's network. Writes a report to
# tio-runs/logs/preflight-<time>.txt and prints suggested sbatch options at the end.
#   bash scripts/exp900/vulcan_preflight.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p tio-runs/logs
REPORT=tio-runs/logs/preflight-$(date +%Y%m%d-%H%M%S).txt
exec > >(tee "$REPORT") 2>&1
section() { printf '\n== %s\n' "$*"; }
mask() { sed -E 's#//[^:/@]+:[^@]+@#//***:***@#g'; }

section host
echo "$(hostname)  user $USER  $(date -Is)  cwd $PWD"
[ -n "${SLURM_JOB_ID:-}" ] && echo "WARNING: inside a Slurm job; run this on a login node"

section git
git fetch -q origin 2>/dev/null || echo "WARNING: git fetch failed (no internet on this node?)"
echo "branch $(git branch --show-current)  HEAD $(git rev-parse --short HEAD)  origin $(git rev-parse --short origin/exp/vulcan-900 2>/dev/null)"
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/exp/vulcan-900 2>/dev/null)" ] || echo "ACTION: git reset --hard origin/exp/vulcan-900   (no local work exists yet, so this only syncs)"
[ -z "$(git status --porcelain)" ] && echo "working tree clean" || { echo "working tree has changes:"; git status --short | head; }

section transferred data
python3 scripts/exp900/manifest.py check
ls tio-runs/data900/three-tier-900-sft-v1/episode_export/annotations.private.jsonl >/dev/null 2>&1 && echo "900 export unzipped" \
  || echo "ACTION: (cd tio-runs/data900 && unzip -q dl/900-private-candidates.zip)"

section .env
if [ -f .env ]; then
  perm=$(stat -c %a .env); [ "$perm" = 600 ] && echo ".env present, mode 600" || echo "ACTION: chmod 600 .env (now $perm)"
  echo "keys set: $(grep -E '^[A-Z_]+_API_KEY=.+' .env | cut -d= -f1 | tr '\n' ' ')"
  grep -E '^TIO_GENERATOR_(PROVIDER|MODEL)=' .env
else
  echo "MISSING .env (copy it from Nibi; it is not in git)"
fi

section local caches from the transfer
if [ -d .venv ]; then
  if .venv/bin/python -c 'import sys' >/dev/null 2>&1 && grep -q "$PWD/.venv" .venv/bin/activate; then echo ".venv usable"
  else echo "ACTION: rm -rf .venv   (built on Nibi, not usable here; scripts/bootstrap_login.sh rebuilds it)"; fi
else echo ".venv absent (scripts/bootstrap_login.sh builds it)"; fi
[ -d .cache/ms-playwright ] && echo "playwright browsers present" || echo "playwright browsers absent (bootstrap downloads them)"
[ -f web/dist/harness.js ] && echo "web build present" || echo "web build absent (bootstrap builds it)"

section modules named in scripts/setup_env.sh
mods=$(grep -E '^module load' scripts/setup_env.sh | sed 's/module load //')
for m in $mods; do
  if module -t spider "$m" >/dev/null 2>&1; then echo "ok   $m"; else echo "MISSING $m -> available: $(module -t spider "${m%%/*}" 2>&1 | grep -E "^${m%%/*}/" | tr '\n' ' ')"; fi
done
echo "CPU arch here: ${RSNT_ARCH:-unknown}  (StdEnv picks x86-64-v3/v4 builds per node; jobs rebuild their venv, so this is informational)"

section wheels pinned by requirements.txt and scripts/exp900/requirements-900.txt
if command -v avail_wheels >/dev/null; then
  for pin in $(grep -hvE '^\s*(#|$)' requirements.txt scripts/exp900/requirements-900.txt | sed 's/\s.*//' | sort -u); do
    name=${pin%%==*}; ver=${pin#*==}; [ "$ver" = "$pin" ] && ver=""
    if avail_wheels "$name" ${ver:+--version "$ver"} --python 3.12 2>/dev/null | grep -qi "^${name//-/[-_]}"; then echo "ok   $pin"; else echo "MISSING $pin"; fi
  done
else
  echo "avail_wheels not found (load a StdEnv first: source scripts/setup_env.sh)"
fi

section Slurm accounts and limits
sacctmgr -n -P show assoc user="$USER" format=account,partition,qos,maxjobs,maxsubmit,maxtres,grptres 2>/dev/null | sort -u
for q in $(sacctmgr -n -P show assoc user="$USER" format=qos 2>/dev/null | tr ',' '\n' | sort -u); do
  sacctmgr -n -P show qos "$q" format=name,maxwall,maxtrespu,maxjobspu,maxsubmitpu,grptres 2>/dev/null
done
sshare -U -u "$USER" 2>/dev/null | tail -n +3
section partitions, GPU types and node states
echo "GPU partitions:"; sinfo -o '%P|%a|%l|%D|%G' 2>/dev/null | grep -v '(null)' | column -t -s '|' | head -20
echo "CPU partitions: $(sinfo -h -o '%P %l %G' 2>/dev/null | grep '(null)' | awk '{printf "%s(%s) ", $1, $2}' | cut -c1-400)"
echo "GPU nodes by state: $(sinfo -h -o '%G %T %D' 2>/dev/null | grep -i gpu | awk '{s[$2]+=$3} END{for(k in s) printf "%s=%d ", k, s[k]}')"
squeue -u "$USER" -h -o '%i %j %T %M %R' 2>/dev/null | head

section login-node network
env | grep -iE '^(https?|no|all)_proxy=' | mask || echo "no proxy variables on the login node"
PROVIDER=$(grep '^TIO_GENERATOR_PROVIDER=' .env 2>/dev/null | cut -d= -f2)
case $PROVIDER in anthropic) VAR=ANTHROPIC_BASE_URL;; openai) VAR=OPENAI_BASE_URL;; openai_compatible) VAR=COMPATIBLE_BASE_URL;; gemini) VAR=GEMINI_BASE_URL;; *) VAR=;; esac
GEN_HOST=$(grep "^$VAR=" .env 2>/dev/null | sed -E 's#^[^=]*=https?://([^/:]+).*#\1#')
for target in ${GEN_HOST:+"https://$GEN_HOST/"} https://huggingface.co/ https://github.com/; do
  label=$target; [ "$target" = "https://$GEN_HOST/" ] && label="generator host ($VAR)"
  printf '%-34s %s\n' "$label" "$(curl -sS -o /dev/null -m 15 -w 'HTTP %{http_code}' "$target" 2>&1 | tail -1)"
done
echo "allow-host for scripts/exp900/connect_proxy.py: the host part of $VAR in .env"

section suggested next steps
ACCT=$(sacctmgr -n -P show assoc user="$USER" format=account 2>/dev/null | grep -m1 '^aip-')
GRES=$(sinfo -h -o '%G' 2>/dev/null | grep -o 'gpu:[a-z0-9_]*' | sort -u | head -3 | tr '\n' ' ')
echo "export ACCT=${ACCT:-<aip-account>}      # GPU types seen: ${GRES:-unknown}"
echo "export TIO_RUNS=$PWD/tio-runs TIO_MODELS=$PWD/models"
echo "CPU_JOB=\"--account=\$ACCT --output=\$TIO_RUNS/logs/%x-%j.out --cpus-per-task=4 --mem=16G\""
echo "GPU_JOB=\"--account=\$ACCT --output=\$TIO_RUNS/logs/%x-%j.out --gpus-per-node=l40s:1 --cpus-per-task=8 --mem=64G\""
echo "then: bash scripts/bootstrap_login.sh, and sbatch \$CPU_JOB --time=00:20:00 scripts/slurm/vulcan_probe.sbatch"
echo "report saved to $REPORT"
