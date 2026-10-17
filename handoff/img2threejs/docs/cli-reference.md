# CLI reference

下列五个入口是当前文档支持的命令。除 Streamlit 外，仓库脚本统一使用 `python -m scripts.<module>`，并从仓库根目录运行。

## Chatbox Viewer

```bash
streamlit run app.py
```

读取 `data/sessions/human_demo/` 下的 Chatbox session、`data/sessions/agent_demo/` 下的版本化 Pipeline trajectory、可选 `data/bundles/`，以及本地 `data/runs/` 下的 Pipeline run。

## Runtime smoke

```bash
python -m scripts.runtime_smoke HTML \
  [--out DIR] \
  [--actions ACTION ...] \
  [--headed] \
  [--chromium EXECUTABLE]
```

- `HTML`：已有完整 HTML。
- `--out`：默认 `data/runs/runtime-smoke`，目录可已存在。
- `--actions`：默认 `orbit_right zoom_in`；可用相对动作由 runtime action parser 定义。
- 首次 render-check 截图为 `render.png`，动作截图为 `01_<action>.png`、`02_<action>.png` 等。
- render check 失败返回 1；通过并完成动作返回 0。
- 复用 `--out` 可能保留或覆盖同名旧文件，正式记录应使用新目录。

## Codegen + render retry

```bash
python -m scripts.run_codegen REFERENCE \
  [--run-id ID] \
  [--headed] \
  [--chromium EXECUTABLE]
```

- 未指定 `--run-id` 时使用 `codegen-YYYYMMDD-HHMMSS`。
- 输出根目录由 `RUNS_DIR` 决定，默认 `data/runs`。
- run 目录必须不存在；该入口不续跑、不覆盖。
- render 成功返回 0，耗尽 render retry 返回 1。

## Active-view Verifier

```bash
python -m scripts.run_verifier REFERENCE HTML \
  [--out DIR] \
  [--budget N] \
  [--action-space relative_discrete_v1|pose_grid_v2] \
  [--action-space-config PATH] \
  [--headed]
```

- `--out` 默认 `data/runs/verifier-smoke`，目录可已存在。
- 正整数 `--budget` 覆盖 `VERIFIER_ACTION_BUDGET`；传入 `0` 会被当前 CLI 当作未设置，负数会在 episode 初始化时失败。
- CLI action-space 参数优先于 `.env`。
- V2 candidate 必须实现 camera harness，否则明确失败，不回退 V1。
- 正常获得 `accept` 或 `revise` 都返回 0；调用方应读取 `verifier_agent.json.verdict`。

## Full iterative Pipeline

```bash
python -m scripts.run_pipeline REFERENCE \
  [--run-id ID] \
  [--headed] \
  [--chromium EXECUTABLE] \
  [--max-visual-revisions N] \
  [--action-space relative_discrete_v1|pose_grid_v2] \
  [--action-space-config PATH]
```

- 未指定 `--run-id` 时使用 `pipeline-YYYYMMDD-HHMMSS`。
- run 目录必须不存在；该入口不续跑、不覆盖。
- `--max-visual-revisions` 覆盖 `.env`，必须不小于 0。
- action-space CLI 参数优先于 `.env`。
- `render_failed` 返回 1；`accepted` 和 `max_visual_revisions` 都返回 0。
- 最终语义必须读取 `summary.json.status` 和 `summary.json.accepted`。

## 路径与覆盖规则

| 入口 | 输出选择 | 已存在目录 |
|---|---|---|
| Viewer | 读取 `data/sessions`、`data/runs` 与可选 bundles | 只读 |
| Runtime smoke | `--out` | 允许，可能混入/覆盖文件 |
| Codegen | `RUNS_DIR/<run-id>` | 拒绝 |
| Verifier | `--out` | 允许，可能混入/覆盖文件 |
| Pipeline | `RUNS_DIR/<run-id>` | 拒绝 |

本文档只描述当前受支持接口；旧 Codex rollout JSONL 不属于受支持输入。
