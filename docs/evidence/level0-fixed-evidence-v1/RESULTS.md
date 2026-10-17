# Level 0 fixed-evidence plus 2B diagnosis replay

This run is a replay of the `35dd59f` Level 0 binary-diagnosis experiment,
not a new SFT run.  It uses the same base checkpoint, LoRA adapter, prompt,
processor, source commit, opaque 3/2 asset split, and human-reviewed labels.
No frozen Test data was touched.

## Fixed evidence protocol

For every one of the ten opaque cases, Chromium loaded the frozen reference
and candidate programs separately.  Each program began with `reset_joints`,
then the fixed policy requested the selected view at rest, actuated the
pre-specified joint once to `limit`, and requested that same view again.  The
four images were composed without resizing as: reference rest, reference
limit, candidate rest, candidate limit.  The diagnosis prompt was the exact
`35dd59f` prompt and the model input contained no asset ID, joint ID, camera,
program parameters, or Gold field.

The execution-only recipe came from the exact source commit's private
`AssetSpec` and frozen screening recipe.  It was not sent to either model and
the scorer used `human_review.jsonl`; `gold.jsonl` was not read by the replay
or scorer.

The Chromium run used the existing `ObservationSession` backed by sibling
`TurnItOver`'s built harness and installed Chromium.  It recorded browser/GL
details, returned camera payloads, returned limit values, per-tile SHA-256s,
and four-panel image comparisons in the ignored local artifact directory.

## Equivalence checks

All ten Chromium-composed 542x542 PNGs were byte-identical to the associated
input image from the prior offline run.  Every tile was 256x256, ordering was
unchanged, reference and candidate used equal returned rest-camera payloads,
and every returned action value equaled the loaded joint's `limit` detent.
Consequently Qwen preprocessing was unchanged (`image_grid_thw=[1,34,34]`,
`pixel_values=[1156,1536]`) and the semantic per-case generations are
unchanged (the fresh records naturally have different wall-clock timings).

## Results

| Method | Train (6) | Held-out (4) | Change from `35dd59f` |
| --- | ---: | ---: | --- |
| Base Qwen3-VL-2B | 3/6 | 2/4 | none |
| `35dd59f` LoRA | 6/6 | 3/4 | none |
| Pixel temporal-delta baseline | 6/6 | 4/4 | none |

The LoRA's only held-out error remains opaque case `l0-2bac52a22825` (false
`pass`); the other held-out failure remains correct.  Because all inputs and
generations match byte-for-byte, no prediction changed for an execution-side
reason.  This establishes a fixed-evidence execution/diagnosis closed loop;
it does **not** show that the model learned to choose observations.

## Reproduction

The local evidence and result artifacts are under
`output/level0-fixed-evidence-v1/`.  Render with the existing browser
environment, then score using the original evaluator and adapter:

```bash
PYTHONPATH=. PLAYWRIGHT_BROWSERS_PATH=../TurnItOver/.cache/ms-playwright \
  ../TurnItOver/.venv-local/bin/python scripts/replay_level0_fixed_evidence.py \
  --source-dataset data/partnet-mobility-level0-axis-v1 \
  --private-source <exact-commit-private-replay-dir> \
  --harness ../TurnItOver/web/dist --esbuild ../TurnItOver/web/node_modules/.bin/esbuild \
  --views configs/views.yaml --out output/level0-fixed-evidence-v1/data
```

The independent audit is `scripts/audit_level0_fixed_evidence.py`; it checks
replayed-image identity, split identity, and all five prediction files against
the `35dd59f` baseline.
