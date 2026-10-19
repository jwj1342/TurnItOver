#!/bin/bash
set -e
cd /home/laq/experiments/partnet-action-sft-v1/code
export PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
exec /home/laq/experiments/kimi-v4-increment-v1/venv/bin/python scripts/evaluate_action_sft_learning.py --dataset ../data/partnet-action-distributions-v1 --model /home/laq/experiments/kimi-v4-increment-v1/models/Qwen3-VL-2B-Instruct --adapter ../output/train/checkpoint --out ../output/learning
