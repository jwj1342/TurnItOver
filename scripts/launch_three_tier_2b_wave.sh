#!/usr/bin/env bash
# Schedule Base, Evidence, and Ordinary 2B fixed-observation jobs safely.
set -euo pipefail

root=/home/laq/TurnItOver-900-base2b
python_bin=/home/laq/experiments/kimi-v4-increment-v1/venv/bin/python
dataset="$root/data/three-tier-900-sft-v1"
model=/home/laq/models/Qwen3-VL-2B-Instruct
manifest_sha=6d0fc5344810396e9454e83c58572f9a9ecdcae4cb53a477f59e9dd541fc6c25
run_id=three-tier-900-2b-20261106-0240
run_root="$root/results/$run_id"
log_root="$root/logs/$run_id"
lock_root="$root/.gpu-locks"

run_worker() {
  local name=$1
  shift
  mkdir -p "$log_root" "$lock_root"
  while true; do
    for gpu in 0 1; do
      set +e
      (
        flock -n 9 || exit 75
        local_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
        local_used=${local_used// /}
        [[ "$local_used" =~ ^[0-9]+$ && "$local_used" -lt 500 ]] || exit 75
        echo "$(date -Is) acquired GPU $gpu" | tee -a "$log_root/$name.log"
        CUDA_VISIBLE_DEVICES="$gpu" "$@" 2>&1 | tee -a "$log_root/$name.log"
      ) 9>"$lock_root/gpu-$gpu.lock"
      status=$?
      set -e
      if [[ $status -ne 75 ]]; then
        return "$status"
      fi
    done
    sleep 60
  done
}

launch_workers() {
  mkdir -p "$run_root" "$log_root" "$lock_root"
  screen -dmS "$run_id-base" bash "$0" --worker base-validation \
    "$python_bin" "$root/scripts/evaluate_qwen_episode_sft.py" \
    --dataset "$dataset" --dataset-manifest-sha256 "$manifest_sha" \
    --model "$model" --out "$run_root/base-validation" --task diagnosis \
    --split validation --seed 20261013 --max-new-tokens 512 --bootstrap-samples 1000 --device cuda
  screen -dmS "$run_id-evidence" bash "$0" --worker evidence-diagnosis \
    "$python_bin" "$root/scripts/train_qwen_episode_sft.py" \
    --dataset "$dataset" --dataset-manifest-sha256 "$manifest_sha" \
    --model "$model" --out "$run_root/evidence-diagnosis" --task diagnosis \
    --supervision evidence --seed 20261013 --epochs 3 --learning-rate 2e-4 \
    --lora-rank 8 --lora-alpha 16 --device cuda
  screen -dmS "$run_id-ordinary" bash "$0" --worker ordinary-diagnosis \
    "$python_bin" "$root/scripts/train_qwen_episode_sft.py" \
    --dataset "$dataset" --dataset-manifest-sha256 "$manifest_sha" \
    --model "$model" --out "$run_root/ordinary-diagnosis" --task diagnosis \
    --supervision ordinary --seed 20261013 --epochs 3 --learning-rate 2e-4 \
    --lora-rank 8 --lora-alpha 16 --device cuda
}

if [[ ${1:-} == --worker ]]; then
  shift
  run_worker "$@"
  exit $?
fi

[[ $# -eq 1 ]] || { echo "usage: $0 YYYY-MM-DDTHH:MM:SS" >&2; exit 2; }
target_epoch=$(date -d "$1" +%s)
now_epoch=$(date +%s)
(( target_epoch > now_epoch )) || { echo "launch time must be in the future" >&2; exit 2; }
sleep "$((target_epoch - now_epoch))"
launch_workers
