#!/bin/bash
# Submit one repair-table run against in-cluster vLLM generator replicas: every judge in JUDGES x NSHARD shards of the
# eligible case list, shards assigned round-robin over the replicas. Output: $ROOT/<judge>/<generator>/<judge>/<case>/,
# job list in $ROOT/jobs.txt. Resubmitting the same command reruns only missing, incomplete and generator_error episodes.
# Usage (repo root):
#   ROOT=tio-runs/repair/q38/rep2 TAG=q38r2 JUDGES="none pixel gold" REPLICAS="<vllm job ids>" bash scripts/exp900/submit_repair.sh
# Optional: JUDGE_SEED (ours_* adapters, default 20261013), NSHARD (jobs per judge, 6), PARALLEL (runner processes per job,
#   1; GPU jobs with PARALLEL>1 share one L40S: up to 4 with 2B judges, 2 with 8B judges), MODEL_ID (Qwen3.8-27B-FP8),
#   MAX_TOKENS (32768).
set -euo pipefail
: "${ROOT:?}" "${TAG:?}" "${JUDGES:?}" "${REPLICAS:?}"
NSHARD=${NSHARD:-6}
P=${PARALLEL:-1}
PACK=tio-runs/data900/repair58/three-tier-900-validation-repair-58-v1
mkdir -p "$ROOT/shards"
grep -v '^#' tio-runs/repair/eligible.txt | awk -v n="$NSHARD" -v d="$ROOT/shards" '{print > (d "/n" n "-" (NR-1) % n ".txt")}'
reps=($REPLICAS)
k=0
for j in $JUDGES; do
  case $j in
    zs8b_*|ours_*) RES="--gpus-per-node=l40s:1 --cpus-per-task=$((P > 1 ? 16 : 8)) --mem=$((P > 1 ? 96 : 64))G" ;;
    *) RES="--cpus-per-task=$((4 * P)) --mem=$((16 * P))G" ;;
  esac
  for s in $(seq 0 $((NSHARD - 1))); do
    rep=${reps[$((k % ${#reps[@]}))]}
    k=$((k + 1))
    ep=$(cat "tio-runs/vllm/$rep.endpoint")
    host=$(echo "$ep" | sed -E 's|http://([^:/]+).*|\1|')
    id=$(TIO_GENERATOR_PROVIDER=openai_compatible TIO_GENERATOR_MAX_TOKENS=${MAX_TOKENS:-32768} TIO_GENERATOR_TIMEOUT=1800 \
         COMPATIBLE_API_KEY=$(cat ~/.tio-vllm-token) TIO_GENERATOR_BASE_URL=$ep GEN_NO_PROXY=$host \
         CASES=$PACK OUT=$ROOT/$j JUDGES=$j GENERATOR=api GENERATOR_MODEL=${MODEL_ID:-Qwen3.8-27B-FP8} ROUNDS=3 \
         CASE_IDS_FILE=$ROOT/shards/n$NSHARD-$s.txt JUDGE_SEED=${JUDGE_SEED:-20261013} PARALLEL=$P \
         sbatch --parsable --account=aip-zhouyang $RES --time=08:00:00 --job-name="$TAG-$j-s$s" \
           --output=tio-runs/logs/%x-%j.out scripts/slurm/repair_table.sbatch)
    echo "$id $j s$s $rep" >> "$ROOT/jobs.txt"
  done
done
tail -n $((k)) "$ROOT/jobs.txt"
