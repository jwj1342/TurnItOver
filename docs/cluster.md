# Running on the Alliance cluster (Nibi)

## Modules and environment

All modules are loaded by `scripts/setup_env.sh` (`module purge; module load StdEnv/2023 python/3.12 nodejs/24.15.0`).
Source it; never execute it; never put module loads in `~/.bashrc`. It also sets `PLAYWRIGHT_BROWSERS_PATH`
to `.cache/ms-playwright` inside the repo and picks the venv (`.venv` on a login node, `$SLURM_TMPDIR/venv`
inside a job).

## Python packages

Only the Alliance wheelhouse is used (`pip install --no-index -r requirements.txt`). Check availability with
`avail_wheels <pkg> --python 3.12` before adding a dependency, then pin it in `requirements.txt`.

## One-time bootstrap (login node, needs internet)

```
bash scripts/bootstrap_login.sh
```

Creates `.venv`, installs wheels (the package itself is used in place via `PYTHONPATH`, set by `setup_env.sh`), downloads `chromium-headless-shell` into `.cache/ms-playwright`, runs
`npm ci` and builds `web/dist`. The browser itself is never launched on the login node; `smoke_browser.sbatch` does that. Compute nodes have no internet: they
only read these directories from scratch. Scratch purges files unused for 60 days; re-run the bootstrap to
restore them.

## Jobs

Everything that runs a browser goes through Slurm (account `def-zhouyang`):

```
sbatch scripts/slurm/smoke_browser.sbatch                 # browser check + unit + browser tests
CONFIG=configs/generate_toy.yaml sbatch scripts/slurm/generate_array.sbatch   # 8 shards
sbatch scripts/slurm/run_cmd.sbatch python -m turnitover detectability --config configs/detectability_toy.yaml --out data/runs/detect_toy.json
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

`run_cmd.sbatch` runs any single command inside the project environment on a compute node; use it for
one-off CLI invocations instead of running them on the login node.

Each array task rebuilds the venv in `$SLURM_TMPDIR` from the wheelhouse (seconds), then generates one
shard. A shard whose manifest says `complete` is skipped, so re-submitting a failed array is safe.
Logs land in `logs/`.

If a compute node lacks the shared libraries headless Chromium needs (`scripts/check_browser.py` fails
at launch), the fallback is an Apptainer image; only `RenderConfig.browser_args`/launcher would change.

## Storage

- Code: this repo on scratch (git). Commit often; scratch is not backed up.
- Generated data: `data/runs/<run_id>/shard-XXXX.{tar,index.jsonl,manifest.json}` — three files per shard
  regardless of sample count. `/project` is near its file-count quota; never expand tars there.
- Check `diskusage_report` before large runs.
