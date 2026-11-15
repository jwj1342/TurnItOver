#!/usr/bin/env bash
# Withheld-evidence evaluation: her evaluator, unchanged, on the derived export where every exposing cell is removed.
# Usage: bash run_withheld.sh <RUN_DIR> <DATASET_DIR> <MODELS_DIR> <SIZE>
set -uo pipefail
TIO_RUNS=${TIO_RUNS:-/scratch/jwj/tio-runs}
RUN=$1; DATASET=$2; MODELS=$3; SIZE=$4
MANIFEST_SHA=$(sha256sum "$DATASET/manifest.json" | cut -d' ' -f1)
STATUS=$RUN/status.tsv; mkdir -p "$RUN/logs"
step() { local name=$1; shift; local t0=$(date +%s)
  if "$@" > "$RUN/logs/$name.log" 2>&1; then printf '%s\tok\t%ss\n' "$name" "$(( $(date +%s) - t0 ))" | tee -a "$STATUS"
  else printf '%s\tFAIL\t%ss\n' "$name" "$(( $(date +%s) - t0 ))" | tee -a "$STATUS"; tail -n 12 "$RUN/logs/$name.log"; fi; }
common=(--dataset "$DATASET" --dataset-manifest-sha256 "$MANIFEST_SHA" --task diagnosis --split validation --seed 20261013 --max-new-tokens 512 --device cuda)
M=$MODELS/Qwen3-VL-$SIZE-Instruct
cd "$TIO_RUNS/code900"
step "$SIZE-base-withheld" python scripts/evaluate_qwen_episode_sft.py "${common[@]}" --model "$M" --out "$RUN/$SIZE/base-withheld"
for ck in "$TIO_RUNS"/out/level900-sft-$SIZE-*/$SIZE/seed*/*-diagnosis/checkpoint; do
  [ -f "$ck/adapter_config.json" ] || continue
  seed=$(basename "$(dirname "$(dirname "$ck")")"); sup=$(basename "$(dirname "$ck")" | cut -d- -f1)
  A=$SLURM_TMPDIR/adapters/$SIZE-$seed-$sup; mkdir -p "$A"; cp "$ck"/* "$A"/
  echo "$ck" > "$A/SOURCE.txt"
  step "$SIZE-$seed-$sup-withheld" python scripts/evaluate_qwen_episode_sft.py "${common[@]}" --model "$M" --adapter "$A" --out "$RUN/$SIZE/$seed-$sup-withheld"
done
echo "RUN DONE $(date -Is)" | tee -a "$STATUS"
