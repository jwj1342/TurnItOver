# v1.1 diagnosis, repair and observation reproduction

This delivery accompanies Issue #10. It includes the [Table 2/3 package](../table23-reproduction/README.md) unchanged and adds the complete five-condition Qwen3.8-Flash repair path, its frozen source versions, input preparation, independent scoring, aggregation and input audit. No dataset, case program, image, adapter, API credential or server login is included.

## Version identities

| Component | Frozen identity |
| --- | --- |
| Dataset | `three-tier-900-sft-v1.1-pre.1` / `three-tier-900-sft-v1.1` |
| Manifest SHA-256 | `b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b` |
| Training and runtime | `89e07db82890af02ae651dcd8b86082981fb7d1b` |
| Initial repair source | `3e1fbd5e0c38afd78ac53548172f4b0f61f06242` on `fix/table1-openhands-protocol` |
| Table 2/3 delivery | `3ca533a72f972cadc82da19e28deead94865bdcc` |
| 44 sorted case IDs, LF terminated | `b628ea8767380800c8af2548000bf8f00d6523557a80716f6dc89c3b307b9c0c` |

The historical branch tip can move. Use the commits above, plus the exact final session files in `provenance.json`; the initial repair commit alone is insufficient. `frozen/` retains original bytes, including v2/v3/v4 workers and Gold implementations. The public wrapper uses final v4 Gold and geometry guards for new runs. It adapts paths and makes the original DNS override optional; the inference wrapper changes only model/adapter path selection. It does not change model prompts, finding predicates, view budgets, geometry thresholds or bootstrap formulas. It rejects a different generator name: another strong model requires a separately frozen protocol and is not already implemented as an experiment here.

The 660 published trajectories used the archived versions and uniform technical reruns described in [RESULTS.md](RESULTS.md). New API runs are stochastic and cannot recreate those trajectories byte for byte. Use the archive aggregation command to verify published measurements. Do not treat a new run as the original artifact.

## Data and scope

Train uses all 600 cases. Validation admits 104: 60 clean and 44 defective (19 Easy, 15 Medium, 10 Hard defective). Sixteen validation injections remain quarantined. The 180 Test cases are not used by these entry points. v1.1 is an experimental prerelease, not a fully human-reviewed 900-case benchmark. Ordinary and Evidence training must be rerun for v1.1; v1 checkpoints are not interchangeable.

Private inputs are obtained through the collaborator's existing access:

* The extracted v1.1 dataset, including train/validation episodes, annotations and images.
* `three-tier-900-validation-repair-58-v1`, the immutable source-program carrier, including `cases.json`, `SHA256SUMS`, and each selected program's TypeScript and spec. It is filtered by v1.1 admission; its name does not make the experiment a v1-label run.
* For Ordinary only: the Qwen3-VL-2B-Instruct base, the v1.1 Ordinary seed `20261013` adapter, and its original `run.json`. The other conditions do not load this adapter.

Preparation checks manifest-bound records, the exact 44 IDs/source objects, image hashes and source program identities. Input paths remain in the private output directory. Always use a new work directory outside the source package; it receives an ignore-all `.gitignore`. Never commit it. Do not use Python `-O` because frozen code uses assertions.

## Environments and rendering

`environment.json` records the observed SOAR package versions. Keep repair and GPU inference environments separate. `requirements-repair.txt` records Python 3.12.3, OpenHands SDK 1.62.0 and Playwright 1.58.0; `requirements-inference.txt` records Python 3.11.15 and the original Qwen stack. Reuse compatible environments where possible. Inference's `torch==2.10.0+cu128` needs the appropriate CUDA wheel source; do not install a GPU stack into the repair environment. These are observed direct dependencies, not a complete transitive lockfile.

Use a UTF-8 locale for archived scripts that read generated program text. On Windows, invoke the Python commands below as `python -X utf8 ...` (or set `PYTHONUTF8=1` before starting Python), so non-ASCII model output is not decoded using the system code page.

Build the pinned repository renderer using its existing Node lockfile:

```bash
cd web
npm ci
npm run build
cd ..
cp sessions/v11-reproduction/frozen/renderer/index.html web/dist/index.html
```

Use `web/dist` as the harness and the installed platform esbuild executable (for example `web/node_modules/@esbuild/linux-x64/bin/esbuild`). The harness hashes in `environment.json` must match. Install the repair environment's Playwright Chromium into a suitable local cache using `python -m playwright install chromium`. A different browser/platform can change pixels; preflight requires every initial image to match exactly and stops on any mismatch. It never drops failed cases to produce a more favorable subset.

Run GPU inference and batch rendering in allocated compute resources under the local cluster rules. These tools do not submit Slurm jobs or connect to any server automatically. Windows tests and the original Linux run are reported separately in [VALIDATION.md](VALIDATION.md).

## Training and Table 2/3

Training remains the repository's `scripts/train_qwen_episode_sft.py --task diagnosis --supervision ordinary|evidence`; use the pinned manifest and the explicit three seeds `20261013`, `20261113`, `20261213`, 3 epochs, lr `2e-4`, rank 8, alpha 16. Do not use the old binary-task parameters. The exact supported CLI is in that script. [Table 2/3 instructions](../table23-reproduction/README.md) provide the 21-evaluation loop per model size and the complete Table 3 preparation/render/replay path. They share the runtime and data identities above.

For each size (set `MODEL` to its local base and `RUNS` to a new output root), in the inference environment:

```bash
for seed in 20261013 20261113 20261213; do
  for supervision in ordinary evidence; do
    python scripts/train_qwen_episode_sft.py --dataset "$DATA" \
      --dataset-manifest-sha256 b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b \
      --model "$MODEL" --out "$RUNS/$supervision-$seed" --task diagnosis \
      --supervision "$supervision" --seed "$seed" --epochs 3 \
      --learning-rate 2e-4 --lora-rank 8 --lora-alpha 16 --device cuda
  done
done
```

## Prepare and preflight repair (no model calls)

From the repository root, set paths for your machine:

```bash
PKG="$PWD/sessions/v11-reproduction"
WORK=/path/to/new-private-repair-work
DATA=/path/to/three-tier-900-sft-v1.1
BUNDLE=/path/to/three-tier-900-validation-repair-58-v1
python "$PKG/repair.py" prepare --dataset "$DATA" --bundle "$BUNDLE" \
  --runtime "$PWD" --harness "$PWD/web/dist" \
  --esbuild "$PWD/web/node_modules/@esbuild/linux-x64/bin/esbuild" --out "$WORK"
python "$PKG/repair.py" preflight --work "$WORK"
```

Preparation copies only the 44 private programs/specs and frozen code into the new workdir. Preflight calibrates Pixel on Train only (600 records; 300 pass / 279 fail fit; 21 uncertain excluded), replays all initial views, checks reference positive controls and candidate failures, and audits initial Gold feedback. Calibration maximizes balanced accuracy with the smallest threshold breaking ties. Gold checks positive controls and reconstructs 54 injection exposure groups (9 rest, 42 motion, 3 unexposed), including the initially equivalent anchor injection without awarding repair credit. A failed check must be investigated; do not alter tolerances or remove cases to bypass it.

## Run five conditions

The generator uses three environment variables, supplied privately by the operator: `TIO_GENERATOR_MODEL=qwen3.8-flash`, `TIO_GENERATOR_BASE_URL` (OpenAI-compatible endpoint ending at the API root), and `TIO_GENERATOR_API_KEY`. Do not place their values in scripts, issues or commits. `TIO_GENERATOR_RESOLVED_IP` is optional and only overrides DNS resolution for that endpoint; it does not disable TLS checks.

Start Ordinary in a separate GPU allocation using the inference environment:

```bash
python "$PKG/serve_ordinary.py" --work "$WORK" \
  --model /path/to/Qwen3-VL-2B-Instruct \
  --checkpoint /path/to/ordinary-20261013/checkpoint \
  --training-record /path/to/ordinary-20261013/run.json
```

The service reads public episode requests from `ordinary-queue-v2`; it checks the training metadata and base config, and records the adapter hash. The provided metadata must come with the corresponding trusted adapter; it is not a cryptographic attestation of training. Stop the service after the run by creating `$WORK/ordinary-queue-v2/stop`. A stale ready/stop file is not automatically removed.

In the repair environment, run each condition (sequentially is supported; API concurrency is an operator resource choice):

```bash
for condition in none pixel ordinary zero-shot-sweep gold; do
  python "$PKG/repair.py" run --work "$WORK" --condition "$condition"
done
```

Every condition schedules the same 44 x 3 grid. `--trials /path/to/private-trials.json` can restrict work to a documented subset of that grid for a smoke test or reviewed retry. Partial results must not be published as full results. Existing complete trials are skipped; incomplete trials are preserved and make the wrapper fail. There is no automatic outcome-based retry or deletion. Archive old attempts and freeze a retry manifest/code version before any rerun; apply a semantic correction uniformly to all affected repetitions, independent of outcome.

The final v4 implementation is used throughout a new batch. Historical v2/v3 are for provenance/review, not for mixing output directories. All generator conditions receive the same full reference/candidate scan (4/6/12 images); candidate images are freshly rendered. Maximum 3 submissions and 10 SDK steps apply to the whole attempt. None continues fixed iterations within that cap; other conditions can stop on public pass or generator finish. Independent checks never control public stop. Invalid diagnostic output becomes uncertain without format retries. Gold gives visible diagnoses, not hidden labels or correct numeric parameters.

## Score, audit and summarize

```bash
python "$PKG/repair.py" score --work "$WORK"
python "$PKG/audit_inputs.py" --work "$WORK" --out /path/to/new-private-input-audit
python "$PKG/summarize.py" --work "$WORK" --out /path/to/new-private-summary
```

Program success remains the final geometry/runtime audit: required part identities, prescribed joint states, per-part Chamfer <= 0.005 m, bbox IoU >= 0.9, and reference runtime-complexity limits. Per-original-injection physical restoration is separate: unsupported rewrites remain null, retain their denominator, and contribute bounds rather than invented labels. The initially physically equivalent anchor injection receives no repair credit.

`summarize.py` runs the exact original aggregation on newly collected records. It requires unique in-protocol trajectories; results are marked complete only when all five sets and physical scores are complete. Bootstrap samples 44 source objects 1,000 times (seed 20261116), retaining all three trajectories per object. Costs to first success are conditional on final success; total recorded pipeline tokens include verifier usage and are not automatically the paper's generated-token metric.

To reproduce published aggregates without calling models, obtain the private four-file archive (`monitor-latest.json`, `selection.private.json`, `measurements.json`, `latest-physical-scores-v1-all-scores.json`) and run:

```bash
python "$PKG/summarize.py" --archive /path/to/private-archive --out /path/to/new-private-recalculation
```

The archive is not shipped publicly. `analysis.json` includes condition counts, confidence intervals, per-finding unknowns/bounds, budget terminations and costs. Dataset and frozen-source validation, trial cardinality and explicit error handling are release checks, not proof that every possible generated program is supported by this restricted Gold parser.

## Remaining work

This package covers the five completed conditions. Full Evidence repair and Qwen3-VL-4B rest/scan baselines are not implemented as completed experiments. The next strong generator is undecided, including whether Kimi concurrency is viable. Image-removal experiments, 8B result reconciliation and paper cost definitions remain separate work. See [RESULTS.md](RESULTS.md) and the associated discussion Issue for protocol differences from Vulcan; this delivery does not merge those success rates or silently adopt its harness.
