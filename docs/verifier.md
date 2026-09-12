# Verifier v1

The verifier loads a candidate Three.js program, acquires evidence under a hard action budget, and
produces a structured verdict with per-defect localization, severity, confidence, evidence references
and suggested repairs. Reference photographs are optional; the task may instead specify runtime requirements.
It is an inference/evaluation harness, not a trained model or a complete generation–repair outer loop.

## Run

No API or weights needed for the offline runtime baseline:

```bash
source scripts/setup_local.sh  # 受管集群改用 scripts/setup_env.sh
python -m turnitover verify --program output/toy-preview/program.ts \
  --policy runtime --budget 5 --out output/verify-runtime
```

Use the configured `.env` judge (OpenRouter supported as `TIO_JUDGE_PROVIDER=openrouter`,
`TIO_JUDGE_MODEL=<exact vision model ID>`, `OPENROUTER_API_KEY=...`):

```bash
python -m turnitover verify --program output/photo-run/program.ts \
  --reference-image reference.jpg --policy active --budget 8 --out output/verify-active
```

`--policy fixed` and `--policy random --seed 42` use the same final judge, with predetermined observations.
The fixed schedule starts with runtime stats, front/oblique views, then joint sweeps and state deltas.
The random baseline shuffles that schedule once with an explicit RNG; it is not an optimal exploration policy.
`--actions request_view` disables actuation and runtime queries for action-type ablations.
`--views custom-views.yaml`, `--max-triangles`, and `--max-draw-calls` specify the task's observation and budget constraints.

The runtime baseline can detect observed triangle/draw-call violations but always returns `uncertain`
when none are found: it does not judge photographs, motion correctness or geometry.

## Local Qwen

Weights live under `models/`, separate from Python package `turnitover/models/`.
The initial small vision baseline is [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
at revision `89644892e4d85e24eaac8bacfd4f463576704203`. This is an upstream pretrained model,
not a verifier fine-tuned on our data. Model size/quality alone does not establish verifier accuracy.

Prepare on a login node, using Alliance wheels only:

```bash
source scripts/setup_local.sh  # 受管集群改用 scripts/setup_env.sh
python3 scripts/download_qwen.py
python -m venv .venv-qwen
.venv-qwen/bin/python -m pip install --no-index -r requirements-qwen.txt
sbatch scripts/slurm/verify_qwen.sbatch \
  --program output/toy-preview/program.ts --reference-image reference.jpg \
  --policy fixed --budget 4 --out output/verify-qwen
```

The Slurm template requests one 20 GB H100 MIG instance, four CPU cores, 24 GB RAM and 20 minutes.
It runs local inference completely offline, with no API key and no automatic weight download.
The client loads once per verifier process, uses bfloat16/SDPA and greedy generation, and caps each
decision at 1536 new tokens. Truncated responses are recorded as errors. The local client supports CUDA;
do not run inference on a login-node CPU. A different NVIDIA GPU allocation may be supplied by overriding
the Slurm resources. Download manifests store revision, sizes and SHA256 hashes.

## Evidence boundary

Both dataset generation and verification use `policy.loop.execute_observation` for actual browser actions.
The existing dataset script loop remains compatible; verifier decisions have their own explicit terminal contract.
The judge receives task text, reference images, available part/joint/view IDs, runtime thresholds, and
successful observation history. It receives the actual acquired images, in documented reference/step order.
Candidate source, `AssetSpec`, corruption labels and exported gold geometry are not passed to the judge.
Candidate part/joint IDs are exposed as action handles, so this is not a pixels-only protocol; keep this
metadata consistent when comparing baselines. Runtime hierarchy is acquired only via a charged action.

Observations cost one; all attempted browser actions are charged, including failed attempts. Rendering
after actuation uses the current camera and preserves other joint states. No free runtime queries or
candidate screenshots are injected. At K observations the judge gets one final decision call with remaining=0.
Active mode therefore makes at most K+1 model calls; fixed/random make one final model call. Every call's
usage and latency are recorded separately, so equal observation budgets do not imply equal inference cost.

Verdicts: `pass`, `fail`, `uncertain`. Termination reasons are separate: `judge_finished`, `budget_exhausted`,
`invalid_decision`, `model_error`, `observation_error`, `program_error`, or infrastructure/internal errors.
Budget exhaustion never synthesizes a pass. Invalid JSON/actions/findings become an uncertain error result,
without silent repair, retries, invented observations or extra budget. Findings must cite existing successful
steps and use known taxonomy/part IDs, finite [0,1] confidence/severity values and repair text. Missing parts
are described in text with an empty part list. Confidence is model-reported, not calibrated.

## Outputs

- `index.html`: offline evidence gallery, findings and limitations.
- `result.json`: verifier schema version, task/config, verdict, termination, budget, model calls and provenance.
- `feedback.json`: verdict/findings for a future repair generator; never interpret an uncertain/error verdict as success.
- `trajectory.jsonl`, `observations/*.png`: progressively saved successful observations.
- `candidate.ts`, `reference-*`: exact program and reference inputs.
- `model_calls/NNN/`: exact prompts, image ordering, raw text responses, native token usage and latency.

Output directories must be new or empty. Exit code 0 means the verification run completed, **not** that
the candidate passed; read `verdict.status`. Execution/decision errors return nonzero. Metadata writes are atomic.
Partial traces remain available after errors. Full input/prompt/response logs can contain user data but not API credentials.
Invalid decisions include a schema-validation explanation in `error_message`; raw model responses are retained.

## Privileged gold audit

`--reference-program known-good.ts` runs existing geometry/runtime checkers AFTER the judge's verdict in a
separate session and writes `audit.json`. It never changes the judge verdict or enters a model prompt.
Reference framing is shared within the audit. Candidate rendering for the judge uses candidate auto-framing:
global scale relative to a reference program is therefore not a supported judge claim in this version.
The audit needs matching part/joint conventions; unsupported or missing states can produce an audit error.
It is not a calibrated oracle or complete physical/kinematic test suite.

## Remaining research work

See [local smoke-test results](verifier-smoke.md) for actual Qwen trials, including rejected model decisions.

Real-photo evaluation, fine-tuning, instance/state-conditioned oracle trajectories, calibrated uncertainty,
formal coverage and metrics aggregation, and the generation–repair outer loop remain to be implemented.
Current schema validation establishes evidence references, not that a model's interpretation of them is true.
Generated programs still execute in the existing browser harness; this is not a general-purpose hostile-code sandbox.
