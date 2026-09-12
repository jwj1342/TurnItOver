# TurnItOver — repo conventions

Read `docs/architecture.md` first. `RP.md` is the research proposal that this code serves.

## Environment
- 个人电脑与通用环境使用 `scripts/bootstrap_local.sh` / `scripts/setup_local.sh`，允许使用 PyPI；计算集群配置见 `docs/cluster.md`。
- 下列模块和 wheelhouse 规则仅适用于使用 `scripts/setup_env.sh` 的受管集群环境，不是所有合作者的安装要求。
- Modules are loaded in exactly one place: `scripts/setup_env.sh` (`source` it). Never `module load` elsewhere, never in `~/.bashrc`.
- Python deps come from the Alliance wheelhouse only. Before adding a dependency run `avail_wheels <pkg> --python 3.12`, then pin it in `requirements.txt`. No conda, no uv, no PyPI.
- First-time setup on a login node: `bash scripts/bootstrap_login.sh` (venv, browser download, `npm ci`, web build).
- Login node is for editing, unit tests, and smoke runs with `n_samples <= 16`. Anything larger goes through `sbatch scripts/slurm/*.sbatch` (account `def-zhouyang`).
- Compute nodes have no internet: browser binaries live in `.cache/ms-playwright`, esbuild in `web/node_modules`, both on scratch.

## Pinned contracts (changing them requires updating docs/protocol.md, SCHEMA_VERSION and tests)
- `web/src/program_abi.ts` — what a generated program must export.
- `web/src/protocol.ts` ⇄ `turnitover/render/protocol.py` — harness request/response JSON.
- `turnitover/core/*` — AssetSpec, ObjectProgram, Action, Sample.
- `turnitover/taxonomy/defects.yaml` — defect ids are dataset labels. After editing run `make docs`.

## Working rules
- README 的说明与新增面向使用者的文档默认使用中文，命令、路径和模型名保留原始拼写。
- 遵循 DRY、关注点分离、SRP、清晰契约、低耦合高内聚、可扩展与任务状态隔离、可观测与可测试、KISS、YAGNI。具体边界见 `docs/engineering-principles.md`；只为实际共同语义抽取模块，避免预设未来需求。
- `pytest` before every commit (unit tests, no browser). `pytest -m browser` when touching `web/` or `turnitover/render/`.
- Randomness only through an explicit `numpy.random.Generator`. No global state in workers; a shard must be reproducible from `(config, shard index)`.
- One responsibility per module. Checkers consume browser-exported evidence only, never `AssetSpec`.
- Out of scope until the corresponding milestone: model training, RL, the LLM generation loop, real asset converters.
