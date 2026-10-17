#!/usr/bin/env bash
# Run from repository root. All destinations must be fresh; source artifacts stay read-only.
set -euo pipefail
run_root=${1:?Usage: bash docs/evidence/motion-grounded-v2-20261017/reproduce.sh output/FRESH_NAME}
qwen_python=../TurnItOver/.venv-qwen/bin/python
export PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
mkdir -p "$run_root"
for arm in direct intermediate; do
  "$qwen_python" docs/evidence/motion-grounded-v2-20261017/provenance/panels-builder.py --source output/motion-grounded-v2-intermediate --out "$run_root/$arm-panels" --variant "$arm"
  "$qwen_python" docs/evidence/motion-grounded-v2-20261017/provenance/transfer-builder.py --source output/motion-grounded-v2-eval-suite-final --out "$run_root/$arm-transfer" --variant "$arm" --suite
  "$qwen_python" docs/evidence/motion-grounded-v2-20261017/provenance/no-images-builder.py --source "$run_root/$arm-panels" --out "$run_root/$arm-no-images" --variant "$arm" --without-images
  if [[ ${MOTION_PREPARE_ONLY:-0} == 1 ]]; then continue; fi
  CUDA_VISIBLE_DEVICES=${MOTION_GPU:-0} "$qwen_python" -m scripts.train_qwen_motion_diagnosis --dataset "$run_root/$arm-panels" --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --out "$run_root/$arm-model" --supervision evidence --epochs 2 --seed 20261016 > "$run_root/$arm-train.log" 2>&1
  for condition in panels transfer no-images; do
    CUDA_VISIBLE_DEVICES=${MOTION_GPU:-0} "$qwen_python" -m scripts.evaluate_qwen_motion_diagnosis --dataset "$run_root/$arm-$condition" --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --adapter "$run_root/$arm-model/checkpoint" --out "$run_root/$arm-$condition-eval" --max-new-tokens 384 > "$run_root/$arm-$condition-eval.log" 2>&1
    "$qwen_python" -m scripts.score_motion_pixel_control --dataset "$run_root/$arm-$condition" --predictions "$run_root/$arm-$condition-eval/predictions.jsonl" --out "$run_root/$arm-$condition-score"
  done
done
