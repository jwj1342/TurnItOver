#!/usr/bin/env bash
# Run from the repository root. DATASET, SPLIT, MODEL, PYTHON, OUT are explicit.
set -euo pipefail
: "${DATASET:?}" "${SPLIT:?}" "${MODEL:?}" "${PYTHON:?}" "${OUT:?}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$OUT/logs"
common=(--dataset "$DATASET" --split "$SPLIT")
evaluate() {
    local name=$1
    shift
    for cohort in train heldout; do
        "$PYTHON" scripts/evaluate_level0_axis.py "${common[@]}" --model "$MODEL" \
            --cohort "$cohort" --out "$OUT/$name/$cohort" "$@" \
            > "$OUT/logs/$name-$cohort.log" 2>&1
    done
}
arm() {
    local rank=$1 alpha=$2 gpu=$3
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PYTHON" scripts/train_level0_axis_sft.py "${common[@]}" --model "$MODEL" \
        --out "$OUT/rank$rank/train" --seed 20261017 --epochs 3 \
        --learning-rate 2e-4 --lora-rank "$rank" --lora-alpha "$alpha" \
        > "$OUT/logs/rank$rank-train.log" 2>&1
    evaluate "rank$rank/eval" --adapter "$OUT/rank$rank/train/checkpoint"
}
# Parent directories are needed for log filenames containing the evaluation name.
mkdir -p "$OUT/logs/rank4/eval" "$OUT/logs/rank8/eval" "$OUT/logs/base"
"$PYTHON" scripts/evaluate_level0_pixel_baseline.py "${common[@]}" --out "$OUT/pixel" \
    > "$OUT/logs/pixel.log" 2>&1
(arm 4 8 "${GPU4:-0}"; evaluate base) &
pid4=$!
arm 8 16 "${GPU8:-1}" &
pid8=$!
status=0
wait "$pid4" || status=1
wait "$pid8" || status=1
exit "$status"
