# Level 0 Qwen3-VL-2B minimal visual pass/fail SFT

This is a deliberately small process and transfer check, not a statistically
meaningful generalization result. The model received one original four-panel
`inputs/*.png` image and one fixed binary prompt, then returned exactly `pass`
or `fail`. It did not receive a board image, an asset ID, a category, private
parameters, or hidden-label metadata.

## Data and split

- Source: `origin/laq/sft-dataset` commit
  `f34b07c127266b401564592bfe38bb6c9e7121fb`.
- Ten `inputs/` files passed the source-manifest SHA-256 audit. Their unchanged
  sheet dimensions are 542x542 (the four row-major panels contain native
  256x256 renders separated by the source framing).
- `human_review.jsonl` supplies the supervised `pass`/`fail` target.
  `private/gold.jsonl` was used only to verify every human label and form five
  clean/axis pairs; it was never serialized into a model prompt.
- Three whole pairs (6 cases) train and two whole pairs (4 cases) are held out;
  no pair crosses the split. Result records use opaque case IDs only.

The local repository does contain old Kimi-increment supervision for both
held-out asset identities. Therefore this run uses only the upstream base
`Qwen3-VL-2B-Instruct` plus the newly trained adapter, never a previous local
adapter. The held-out result is useful as within-run object-disjoint transfer,
but must not be presented as proof that the underlying base model had never
seen either asset in earlier project data.

## Configuration

One LoRA run used Qwen3-VL-2B-Instruct on RTX 4090 D, seed `20261017`, rank 8,
alpha 16, learning rate `2e-4`, and 12 epochs (72 one-example updates). The
processor retained one image in its original sheet order; its actual visual
batch was `image_grid_thw=[1,34,34]`, `pixel_values=[1156,1536]`. Training
loss was 0.12937 initially, 7.15e-7 at the final step, and 0.09128 mean.
The adapter SHA-256 is
`15be9f1e942b6dd43317cb26ef9080eb18f82f36545aff330525868ddfb9d3af`.

## Results

| Model / method | Train (6) | Held-out (4) |
| --- | ---: | ---: |
| Base Qwen3-VL-2B | 3/6 | 2/4 |
| This run's LoRA adapter | 6/6 | 3/4 |
| Raw-pixel temporal-delta baseline | 6/6 | 4/4 |

The base output `pass` for every case, so it did not identify any of the five
obvious errors. The adapter fit its six examples and changed one held-out
error from false pass to correct fail, but still output pass on the other
held-out error. The raw-pixel baseline computes the mean absolute difference
between reference and candidate temporal RGB deltas at the fixed source panel
positions. It selects one threshold from the train cases only, then obtains
4/4 held-out; this demonstrates that Level 0 is intentionally visually easy,
not that pixel residuals diagnose motion semantics.

All per-case generations, processor shapes, source-image hashes, split,
training loss, adapter checkpoint, and baseline scores are saved below the
ignored local run directory `output/level0-axis-binary-v1/`.

## Reproduction

Run from the repository root with the already validated offline Qwen
environment. All output directories must be fresh.

```bash
PYTHONPATH=. ../TurnItOver/.venv-qwen/bin/python scripts/prepare_level0_axis_sft.py \
  --dataset data/partnet-mobility-level0-axis-v1 --out output/level0-axis-binary-v1/data
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/train_level0_axis_sft.py \
  --dataset data/partnet-mobility-level0-axis-v1 --split output/level0-axis-binary-v1/data/split.json \
  --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --out output/level0-axis-binary-v1/train \
  --seed 20261017 --epochs 12 --learning-rate 2e-4 --lora-rank 8 --lora-alpha 16
```

Use `scripts/evaluate_level0_axis.py` with `--cohort train|heldout` and an
optional `--adapter`, and run `scripts/evaluate_level0_pixel_baseline.py` for
the transparent pixel control.
