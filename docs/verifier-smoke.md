# Local verifier smoke tests

Recorded 2026-09-12. These runs check integration and failure handling, not model accuracy.

## Setup

- Model: `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`.
- Weights: `models/Qwen3-VL-2B-Instruct/`; downloaded weights match the upstream SHA256.
- Runtime: Transformers 4.57.6, Torch 2.7.1, local offline inference on a Slurm H100 20 GB MIG allocation.
- Candidate: `output/verifier-inputs/candidate.ts`, a synthetic articulated cabinet with 6228 triangles against a 5000-triangle limit.
- Reference: `output/toy-preview/screenshots/06.png`, a synthetic render, not a photograph.
- Fixed/active observation budget: 4. Greedy decoding; no automatic correction or retries.

## Results

| Run | Observation policy | Outcome |
| --- | --- | --- |
| `output/verifier-runtime-demo/` | Runtime, budget 5 | Deterministic runtime check reports the triangle-budget violation; separate gold audit completes. |
| `output/verifier-qwen-fixed/` (job 21746750) | Fixed | Qwen adds action fields inside its verdict; strict schema rejects it. |
| `output/verifier-qwen-fixed-v2/` (job 21776172) | Fixed, verdict-only final prompt | Qwen cites the triangle violation at step 0 but labels the overall verdict `pass`; consistency validation rejects it. |
| `output/verifier-qwen-active/` (job 21776240) | Active | Qwen immediately emits a finding citing step 0 before acquiring any observation; evidence validation rejects it. No browser action is executed. |

All three Qwen runs produce `uncertain` with termination `invalid_decision` and a nonzero CLI exit code. Model loading and image inference work, but these samples do not establish a reliable Qwen judge or successful autonomous action selection. The fixed runs use different prompts and are not controlled comparative measurements. All raw responses are retained in each run's `model_calls/` directory.

The active run's apparent triangle finding is unsupported: the model had neither candidate imagery nor runtime statistics. It resembles a prompt example and must not be counted as successful defect detection.

## Reproduce

From the repository root, choose a new output directory:

```bash
sbatch scripts/slurm/verify_qwen.sbatch \
  --program output/verifier-inputs/candidate.ts \
  --reference-image output/toy-preview/screenshots/06.png \
  --policy active --budget 4 --out output/verifier-qwen-next
```

The example inputs and outputs are local, ignored artifacts. See [verifier.md](verifier.md) for setup and running with your own inputs. OpenRouter configuration is supported but no paid remote model was called in these smoke tests.
