# TurnItOver

Data engine and observation harness for **verification as active perception** of LLM-generated 3D programs
(see `RP.md`). Corrupt a known-good articulated object program, observe it through a budgeted set of actions
(views, joint actuation, runtime probes) in headless Chromium, score it with deterministic checkers, and write
perfectly labeled samples.

Status: repository skeleton plus one end-to-end vertical slice on a toy cabinet asset.

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

## Tests

```bash
pytest              # unit tests, no browser
pytest -m browser   # needs web/dist and the Playwright browser
```

## Read next

`docs/architecture.md` for module boundaries and the pinned contracts, `docs/protocol.md` for the harness API,
`CLAUDE.md` for working conventions.
