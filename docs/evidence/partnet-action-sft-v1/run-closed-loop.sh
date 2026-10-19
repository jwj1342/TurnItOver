#!/bin/bash
set -euo pipefail
# From SFT checkout, with both localhost SSH forwards and model servers ready.
# Every output below must be fresh. Original interrupted runs remain excluded.
export PYTHONPATH=. PLAYWRIGHT_BROWSERS_PATH=../TurnItOver/.cache/ms-playwright OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
PYTHON=../TurnItOver/.venv-local/bin/python
COMMON=(--benchmark ../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1
 --oracle ../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4
 --asset-workspace ../TurnItOver-dataset --harness-workspace ../TurnItOver
 --config configs/generate_partnet_mobility_supervision.yaml
 --model /home/laq/experiments/kimi-v4-increment-v1/models/Qwen3-VL-2B-Instruct
 --adapter /home/laq/experiments/partnet-action-sft-v1/output/train/checkpoint --budgets 2 4 6)
"$PYTHON" scripts/evaluate_action_sft_closed_loop.py "${COMMON[@]}" \
 --model-endpoint http://127.0.0.1:8893 --modes zero_shot fixed random \
 --out output/partnet-action-sft-v1/baselines-run3 &
BASE_PID=$!
"$PYTHON" scripts/evaluate_action_sft_closed_loop.py "${COMMON[@]}" \
 --model-endpoint http://127.0.0.1:8894 --modes action_sft \
 --out output/partnet-action-sft-v1/action-sft &
SFT_PID=$!
wait "$BASE_PID"
wait "$SFT_PID"
"$PYTHON" scripts/action_sft_metrics.py \
 --runs output/partnet-action-sft-v1/baselines-run3 output/partnet-action-sft-v1/action-sft \
 --out output/partnet-action-sft-v1/comparison.json
