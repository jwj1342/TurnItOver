#!/usr/bin/env bash
set -euo pipefail
cd /home/laq/experiments/kimi-v4-increment-v1/code
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
arm="$1"
export CUDA_VISIBLE_DEVICES="$2"
../venv/bin/python scripts/train_qwen_episode_sft.py \
  --dataset "../data/$arm" --model ../models/Qwen3-VL-2B-Instruct \
  --out "../output/$arm-train" --supervision evidence \
  --seed 20261013 --epochs 3 --max-steps 1617 --learning-rate 2e-4 \
  --lora-rank 8 --lora-alpha 16 --diagnosis-weight 1 --stopping-weight 0.2564102564102564
../venv/bin/python scripts/evaluate_qwen_episode_sft.py \
  --dataset ../data/baseline --model ../models/Qwen3-VL-2B-Instruct \
  --adapter "../output/$arm-train/checkpoint" --out "../output/$arm-eval" --max-new-tokens 512
