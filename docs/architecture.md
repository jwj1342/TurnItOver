# Architecture

TurnItOver implements the data engine and observation harness described in `RP.md`: verification of
generated 3D programs treated as *active perception*. This document is the map; `docs/protocol.md` is
the contract; `docs/taxonomy.md` is the label vocabulary.

## Design stance

- The canonical representation on the data-engine side is `AssetSpec` (`turnitover/core/spec.py`).
  Corruptions mutate a spec and re-emit; the Three.js program is *derived* (`turnitover/assets/emit.py`).
- The browser only sees a program satisfying the **Program ABI** (`web/src/program_abi.ts`). The same ABI
  will be what a frontier model is prompted to satisfy, so LLM-written programs flow through the same harness.
- Checkers consume only browser-exported **evidence** (geometry, stats, hierarchy), never the spec. They
  therefore behave identically on spec-derived and LLM-written programs.
- The judge sees only an observation trajectory produced by `policy -> loop -> session`. The data engine,
  the detectability matrix, oracle experiments and future inference all reuse this one seam.

## Data flow

```
AssetSource ──AssetSpec──▶ Corruption ──ObjectProgram(spec, ts)──▶ transpile(esbuild) ──JS──▶ ObservationSession
                                                                                                │  (Playwright ⇄ window.harness)
                       ScriptedPolicy ──Action──▶ run_observation_loop ◀──Observation(png, json)─┘
                                                        │
                                          collect_evidence ──Evidence──▶ Checkers ──CheckResult──┐
                                                                                                 ▼
                                                             Sample ──▶ ShardWriter ──▶ shard-XXXX.{tar,index.jsonl,manifest.json}
```

## Packages

| package | responsibility | depends on |
|---|---|---|
| `core` | pinned dataclasses: spec, program, actions, sample; JSON serde | – |
| `taxonomy` | `defects.yaml` loader, doc renderer | – |
| `assets` | `AssetSource` protocol, toy asset, spec → TypeScript emitter | core |
| `corruptions` | registry keyed by defect id; three implementations | core, taxonomy |
| `render` | Playwright session, TS transpile, view set, protocol mirror | core, checkers.base |
| `policy` | `JudgePolicy` protocol, `ScriptedPolicy`, budgeted loop | core |
| `checkers` | evidence collection, pure mesh math, two checkers | core, render (evidence only) |
| `storage` | shard tar writer/reader | core, checkers.base |
| `engine` | sharding, the vertical slice `run_generate` | everything above |
| `detectability` | (defect, action) matrix on an asset | render, corruptions, checkers.meshops |
| `models` | role config, REST vision adapters, single-call CLI workflows | stdlib, Pillow; syntax check for reconstruction |
| `verifier` | validated decisions, budgeted episodes, active/fixed/random/runtime policies, reports and isolated audits | core, policy execution, render, models |
| `preview`, `media` | offline visual gallery and frame-stepped videos | render, Pillow, system FFmpeg |
| `telemetry`, `config`, `cli` | logging, YAML config, argparse | – |

Rules: `checkers` never imports `assets` or `core.spec`; `policy` never imports `render` (it types the
session structurally); `engine`, `verifier.runner` and CLI export workflows wire their respective tasks together.

`verifier.report` owns offline HTML presentation; `verifier.runner` assembles execution and persistence.
See [工程原则检查](engineering-principles.md) for the current evidence and limitations against the nine engineering principles.

## Statelessness and scale

A shard is fully determined by `(config, shard, n_shards)`: indices come from `shard_indices`, per-sample
seeds from `sample_seed(base_seed, idx)`. Slurm array tasks own one shard each, write three files, and a
completed manifest makes re-submission a no-op. One browser per worker process; it restarts every
`browser_restart_every` loads.

## Pinned contracts

1. Program ABI (`createObject(THREE) -> {root, joints}`, part/pivot naming).
2. `window.harness` methods and JSON payloads, including base64 typed-array geometry export and explicit framing.
3. `AssetSpec` field set.
4. Taxonomy ids `layer.slug`.
5. `Sample` schema (`SCHEMA_VERSION`) and shard tar layout.
6. `Action` types, `Detent` values, view ids, `action_key`.
7. `CheckResult` / `Invariant` shape.
8. The `policy -> loop -> session` seam.

## Deliberately not built (YAGNI)

Training loops and torch deps, RL, the multi-round frontier-LLM repair loop and edit memory, oracle policy,
templated NL diagnosis, real asset converters, non-box parts, textures and material corruptions,
physics / manifold / watertight checkers, HDF5, pydantic, Hydra, structlog, CI, Apptainer (fallback only),
continuous views, a dev web server, partial-shard resume.

`reconstruct` is a single-pass photo-to-program baseline, not a repair loop. `verify` uses an explicit
VerificationPolicy decision contract, sharing browser action execution with the legacy dataset JudgePolicy loop.
Preview orbit views are presentation-only; policy views remain discrete. Optional local Qwen dependencies
are isolated in requirements-qwen.txt; baseline data generation has no torch requirement.
