#!/bin/bash
set -euo pipefail
# Run from the remote experiment's code directory; output must be fresh.
export PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
exec /home/laq/experiments/kimi-v4-increment-v1/venv/bin/python scripts/train_qwen_action_sft.py \
 --dataset ../data/partnet-action-distributions-v1 \
 --model /home/laq/experiments/kimi-v4-increment-v1/models/Qwen3-VL-2B-Instruct \
 --out ../output/train --epochs 3 --seed 20261017 --learning-rate 2e-4
