#!/usr/bin/env bash
# Three-tier-900 fixed-observation diagnosis, her scripts unchanged (code900/scripts/*). Run from $TIO_RUNS/code900.
# Usage: bash run_900.sh <RUN_DIR> <DATASET_DIR> <MODELS_DIR>
#   SPLIT=validation|test (default validation) selects the evaluation split.
#   MODE=sft   SIZE=2B SEEDS="20261013 20261113 20261213"  -> per seed: train evidence + ordinary, eval both on validation
#   (seeds spaced >=3 apart: her trainer shuffles with random.Random(seed + epoch), so consecutive seeds would share data orders)
#   MODE=base  SIZES="2B 4B 8B"                              -> zero-shot eval per size
#   MODE=eval  SIZE=2B SEEDS=... SUPS=...                    -> eval existing checkpoints in $TIO_RUNS/out/level900-sft-$SIZE-* on $SPLIT
set -uo pipefail
TIO_RUNS=${TIO_RUNS:-/scratch/jwj/tio-runs}
SPLIT=${SPLIT:-validation}
RUN=$1; DATASET=$2; MODELS=$3
MANIFEST_SHA=${MANIFEST_SHA:-6d0fc5344810396e9454e83c58572f9a9ecdcae4cb53a477f59e9dd541fc6c25}   # v1: her launch_three_tier_2b_wave.sh
RUN_TAG=${RUN_TAG:-level900}   # checkpoints of MODE=eval come from $TIO_RUNS/out/$RUN_TAG-sft-*
STATUS=$RUN/status.tsv
mkdir -p "$RUN/logs"
step() {
    local name=$1; shift
    local t0=$(date +%s)
    if "$@" > "$RUN/logs/$name.log" 2>&1; then
        printf '%s\tok\t%ss\n' "$name" "$(( $(date +%s) - t0 ))" | tee -a "$STATUS"
    else
        printf '%s\tFAIL\t%ss\n' "$name" "$(( $(date +%s) - t0 ))" | tee -a "$STATUS"
        tail -n 12 "$RUN/logs/$name.log"
    fi
}
common=(--dataset "$DATASET" --dataset-manifest-sha256 "$MANIFEST_SHA")
evalcommon=(--task diagnosis --split "$SPLIT" --seed 20261013 --max-new-tokens 512 --bootstrap-samples 1000 --device cuda)   # eval seed fixed by her protocol
case "${MODE:-sft}" in
  base)
    step pixel-status python "$TIO_ROOT/scripts/exp900/pixel_status_baseline.py" --dataset "$DATASET" --out "$RUN/pixel-status"
    for size in ${SIZES:-2B 4B 8B}; do
        M=$MODELS/Qwen3-VL-$size-Instruct
        step "$size-base-$SPLIT" python scripts/evaluate_qwen_episode_sft.py "${common[@]}" --model "$M" --out "$RUN/$size/base-$SPLIT" "${evalcommon[@]}"
    done ;;
  sft)
    size=${SIZE:-2B}; M=$MODELS/Qwen3-VL-$size-Instruct
    for seed in ${SEEDS:-20261013 20261113 20261213}; do
        for sup in ${SUPS:-evidence ordinary}; do
            T=$RUN/$size/seed$seed/$sup-diagnosis
            step "$size-seed$seed-$sup-train" python scripts/train_qwen_episode_sft.py "${common[@]}" --model "$M" --out "$T" \
                --task diagnosis --supervision "$sup" --seed "$seed" --epochs 3 --learning-rate 2e-4 --lora-rank 8 --lora-alpha 16 --device cuda
            [ -f "$T/checkpoint/adapter_config.json" ] || continue
            step "$size-seed$seed-$sup-eval-$SPLIT" python scripts/evaluate_qwen_episode_sft.py "${common[@]}" --model "$M" --adapter "$T/checkpoint" \
                --out "$T/eval-$SPLIT" "${evalcommon[@]}"
        done
    done ;;
  eval)
    size=${SIZE:-2B}; M=$MODELS/Qwen3-VL-$size-Instruct
    for seed in ${SEEDS:-20261013 20261113 20261213}; do
        for sup in ${SUPS:-evidence ordinary}; do
            for ck in "$TIO_RUNS"/out/$RUN_TAG-sft-$size-*/$size/seed$seed/$sup-diagnosis/checkpoint; do
                [ -f "$ck/adapter_config.json" ] || continue
                step "$size-seed$seed-$sup-eval-$SPLIT" python scripts/evaluate_qwen_episode_sft.py "${common[@]}" --model "$M" --adapter "$ck" \
                    --out "$RUN/$size/seed$seed/$sup-diagnosis/eval-$SPLIT" "${evalcommon[@]}"
            done
        done
    done ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac
echo "RUN DONE $(date -Iseconds)" | tee -a "$STATUS"
