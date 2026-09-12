# TurnItOver

Data engine and observation harness for **verification as active perception** of LLM-generated 3D programs
(see `RP.md`). Corrupt a known-good articulated object program, observe it through a budgeted set of actions
(views, joint actuation, runtime probes) in headless Chromium, score it with deterministic checkers, and write
perfectly labeled samples.

Status: toy data engine, visual exports, role-based model API adapters, a single-pass photo-to-program baseline,
and a budgeted verifier with active/fixed/random VLM policies and an offline runtime baseline.
No external reconstruction pipeline or fine-tuned judge has been reproduced yet.

## Layout

```
turnitover/   Python package: core contracts, taxonomy, assets, corruptions, render, policy, checkers, storage, engine
web/          TypeScript: Three.js runtime + window.harness (built with esbuild into web/dist)
configs/      views.yaml, generate_toy.yaml, detectability_toy.yaml
scripts/      setup_env.sh, bootstrap_login.sh, check_browser.py, slurm/*.sbatch
tests/        unit (no browser) and browser (marked) tests
docs/         architecture.md, protocol.md, taxonomy.md (generated), cluster.md
```

## Quick start (Nibi)

```bash
source scripts/setup_env.sh
bash scripts/bootstrap_login.sh            # once, on a login node
sbatch scripts/slurm/smoke_browser.sbatch  # browser check + tests on a compute node
CONFIG=configs/generate_toy.yaml sbatch scripts/slurm/generate_array.sbatch
python -m turnitover inspect data/runs/toy_smoke/shard-0000.tar --n 4
```

CLI (run inside a job, e.g. via `sbatch scripts/slurm/run_cmd.sbatch <command>` or an `salloc` session):

```bash
python -m turnitover generate --config configs/generate_toy.yaml --shard 0/2 --n-samples 8
python -m turnitover detectability --config configs/detectability_toy.yaml --out data/runs/detect_toy.json
python -m turnitover inspect data/runs/toy_smoke/shard-0000.tar --n 4
python -m turnitover render-docs        # lightweight; fine on a login node
```

## Visual output

Export actual browser renders to `output/` (requires the built web harness and Chromium):

```bash
source scripts/setup_env.sh
python -m turnitover preview
# Or on a compute node:
sbatch scripts/slurm/run_cmd.sbatch python -m turnitover preview
```

Each run creates a timestamped folder containing:

- `index.html`: offline gallery; open locally in a browser after downloading the folder.
- `screenshots/`: 12 rest-pose views and one upper-limit image per joint.
- `overview.png`: labeled contact sheet.
- `turntable.mp4`: 360-degree orbit, 4 seconds by default.
- `joints.mp4`: each joint moving separately, 4 seconds per joint.
- `program.ts`, `manifest.json`: source, camera definitions, framing, joint limits, runtime stats and run metadata.

Videos use FFmpeg with `libx264`, fixed frame steps, and browser-compatible H.264/yuv420p;
their playback speed does not depend on rendering speed. FFmpeg is a system dependency,
not a Python package. Use `--ffmpeg /path/to/ffmpeg` to select it or `--no-video` for PNG-only output.
Useful options: `--out output/my-preview --size 640 --fps 24 --seconds 4`.
`--out` must be new or empty, to preserve previous results. Generated media are gitignored.
Use `--program path/to/object.ts` to preview another program satisfying the Program ABI.
Continuous orbit/motion is for visualization only; the research policy retains its discrete action space.

## Model APIs and photo inputs

Configure `.env` using [.env.example](.env.example), selecting provider and exact model ID independently
for generator, judge and diagnosis. Existing shell variables override `.env`; secrets are gitignored.

```bash
source scripts/setup_env.sh
python -m turnitover models-check   # local configuration only, no API calls
python -m turnitover reconstruct --image reference.jpg --out output/photo-run
python -m turnitover preview --program output/photo-run/program.ts --out output/photo-preview
```

Use `reconstruct ... --dry-run` to prepare inputs without a key or network request.
See [model setup](docs/models.md) for supported APIs, overrides and limitations, and
[pipeline comparison](docs/pipeline-options.md) for actual reconstruction candidates and integration gaps.

## Verifier

```bash
python -m turnitover verify --program output/toy-preview/program.ts \
  --policy runtime --budget 5 --out output/verify-runtime
# Active VLM uses the .env judge role:
python -m turnitover verify --program candidate.ts --reference-image reference.jpg \
  --policy active --budget 8 --out output/verify-active
```

Outputs include an HTML evidence gallery, structured verdict/repair feedback, observation trace and model-call
records. Budget exhaustion, uncertain judgment and execution errors are distinct. Local Qwen3-VL weights go
under `models/`; see [verifier guide](docs/verifier.md) for GPU setup, Slurm commands and research limitations.

## Reproducible examples

```bash
source scripts/setup_env.sh
python scripts/run_examples.py --out output/my-examples
```

This produces two complete synthetic cases (triangle budget and joint-axis corruption), with reference/candidate
screenshots, videos, verifier traces and independent audits. See [experiment guide](docs/experiments.md)
and the [saved example gallery](output/research-examples/index.html). These are integration examples, not a research benchmark.

## Tests

```bash
pytest              # unit tests, no browser
pytest -m browser   # needs web/dist and the Playwright browser
```

## Read next

`docs/architecture.md` for module boundaries and the pinned contracts, `docs/protocol.md` for the harness API,
`CLAUDE.md` for working conventions.
