#!/usr/bin/env bash
#SBATCH --job-name=tio-repair
# Repair table job (scripts/run_repair_table.py). Submit from the repo root; account, GPU (only for zs8b_* and ours_*
# judges), CPU, memory, time and --output on the command line (docs/vulcan-handoff.md).
# Required env: CASES (case dir or pack root), OUT (output dir), JUDGES (space list), GENERATOR (api|scripted-stop|scripted-fix).
# Optional: GENERATOR_MODEL, ROUNDS (3), LIMIT, CASE_IDS (space list), CASE_IDS_FILE (one id per line), CONFIG,
#           GEN_PROXY (http://user:token@host:port, used only for the generator API), PROBE_URL (curl probe at start),
#           GEN_NO_PROXY (comma list of hosts, e.g. the in-cluster vLLM node, that bypass the prolog's http_proxy),
#           TIO_GENERATOR_RETRY_BUDGET_S (seconds of retries per generator call, default 1800),
#           TIO_RUNS (default <repo>/tio-runs), TIO_MODELS (default <repo>/models).
# Exit status is the runner's: non-zero if any episode ended in generator_error (resubmit to rerun those).
set -uo pipefail
: "${CASES:?}" "${OUT:?}" "${JUDGES:?}" "${GENERATOR:?}"
CASE_IDS=${CASE_IDS:-}
REPO=${TIO_ROOT:-$SLURM_SUBMIT_DIR}
TIO_RUNS=${TIO_RUNS:-$REPO/tio-runs}; TIO_MODELS=${TIO_MODELS:-$REPO/models}
cd "$REPO" && source scripts/setup_env.sh        # modules, PLAYWRIGHT_BROWSERS_PATH, PYTHONPATH
virtualenv --no-download "$SLURM_TMPDIR/venv" > /dev/null || exit 1
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index -q -r requirements.txt -r scripts/exp900/requirements-900.txt || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
if [ -n "${GEN_PROXY:-}" ]; then
  export https_proxy=$GEN_PROXY HTTPS_PROXY=$GEN_PROXY         # urllib in turnitover/models/client.py reads these
fi
if [ -n "${GEN_NO_PROXY:-}" ]; then                           # urllib prefers no_proxy, curl reads either
  export no_proxy=${no_proxy:+$no_proxy,}$GEN_NO_PROXY NO_PROXY=${NO_PROXY:+$NO_PROXY,}$GEN_NO_PROXY
fi
[ -n "${PROBE_URL:-}" ] && { curl -sS -o /dev/null -m 15 -w "probe $PROBE_URL HTTP %{http_code}\n" "$PROBE_URL" || echo "probe $PROBE_URL unreachable"; }
mkdir -p "$OUT"
echo "repo $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)" > "$OUT/env-$SLURM_JOB_ID.txt"
# judge weights on node-local disk, only for the judges requested
MODEL_ARGS=()
need() { for j in $JUDGES; do case $j in $1) return 0;; esac; done; return 1; }
if need 'zs8b_*' || need 'ours_*'; then
  mkdir -p "$SLURM_TMPDIR/models"
  sizes=""; need 'zs8b_*' && sizes="8B"; need 'ours_2b_*' && sizes="$sizes 2B"; need 'ours_8b_*' && sizes="$sizes 8B"
  for size in $(echo $sizes | tr ' ' '\n' | sort -u); do cp -r "$TIO_MODELS/Qwen3-VL-$size-Instruct" "$SLURM_TMPDIR/models/" || exit 4; done
  MODEL_ARGS+=(--model-root "$SLURM_TMPDIR/models")
  declare -A SRC=( [ours_2b_evidence]=2B/seed20261013/evidence [ours_2b_ordinary]=2B/seed20261013/ordinary [ours_8b_evidence]=8B/seed20261013/evidence )
  for name in "${!SRC[@]}"; do
    need "$name" || continue
    size=${SRC[$name]%%/*}
    ck=$(ls -d "$TIO_RUNS"/out/level900-sft-$size-*/${SRC[$name]}-diagnosis/checkpoint 2>/dev/null | head -1)
    [ -f "$ck/adapter_config.json" ] || { echo "adapter missing for $name"; exit 5; }
    cp -r "$ck" "$SLURM_TMPDIR/models/$name" && MODEL_ARGS+=(--adapter "$name=$SLURM_TMPDIR/models/$name")
    echo "$name $ck" >> "$OUT/env-$SLURM_JOB_ID.txt"
  done
  python -c "import torch,transformers,peft; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, peft.__version__)" >> "$OUT/env-$SLURM_JOB_ID.txt"
fi
EXTRA=()
CONFIG=${CONFIG:-configs/generate_replicacad.yaml}
PACK=0; [ -f "$CASES/manifest.json" ] && grep -q three-tier-900-validation-repair-programs-v1 "$CASES/manifest.json" && PACK=1
[ "$PACK" = 1 ] && CONFIG=${CONFIG_PACK:-configs/repair_pack900.yaml}
EXTRA+=(--config "$CONFIG")
[ -n "${GENERATOR_MODEL:-}" ] && EXTRA+=(--generator-model "$GENERATOR_MODEL")
[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
[ -n "${CASE_IDS_FILE:-}" ] && CASE_IDS="$CASE_IDS $(grep -v '^#' "$CASE_IDS_FILE" | tr '\n' ' ')"
[ -n "${CASE_IDS// /}" ] && EXTRA+=(--case-ids $CASE_IDS)
# her 58-case pack: labels and per-cell exposure come from the 900 export
EXPORT900=${EXPORT900:-$TIO_RUNS/data900/three-tier-900-sft-v1/episode_export}
[ "$PACK" = 1 ] && EXTRA+=(--labels "$EXPORT900/annotations.private.jsonl" --episodes "$EXPORT900/validation.episodes.jsonl")
python scripts/run_repair_table.py --cases "$CASES" --out "$OUT" --judges $JUDGES --generator "$GENERATOR" \
  --rounds "${ROUNDS:-3}" "${MODEL_ARGS[@]}" "${EXTRA[@]}"
rc=$?
echo "RUN DONE $(date -Is) exit=$rc"
exit $rc
