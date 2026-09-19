# Configuration

Agent 相关入口通过 `Settings.from_env()` 读取仓库根目录 `.env`。环境中已存在的变量不会被 `.env` 覆盖。不要提交包含凭证的 `.env`；仓库只版本化 `.env.example`。

## 模型配置

| 变量 | 默认值 | 约束/语义 |
|---|---:|---|
| `QWEN_BASE_URL` | 空 | Agent 入口必填 |
| `QWEN_API_KEY` | 空 | Agent 入口必填 |
| `QWEN_MODEL` | 空 | Agent 入口必填；OpenAI-compatible 模型通常使用 provider 前缀 |
| `QWEN_MAX_OUTPUT_TOKENS` | `8192` | 每次模型调用的最大输出 token |
| `QWEN_TEMPERATURE` | 空 | 空值表示使用后端默认值，否则解析为浮点数 |

## Agent 与观察预算

| 变量 | 默认值 | 校验 |
|---|---:|---|
| `CODE_AGENT_MAX_STEPS` | `8` | `>= 1` |
| `VERIFIER_AGENT_MAX_STEPS` | `10` | `>= 1` |
| `VERIFIER_ACTION_BUDGET` | `6` | `>= 1`；只计算主动观察动作，初始截图和 finish 不计入 |

模型 step budget 与 observation budget 相互独立。

## Camera action space

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VIEW_ACTION_SPACE` | `relative_discrete_v1` | 可选 `relative_discrete_v1` 或 `pose_grid_v2` |
| `VIEW_ACTION_SPACE_CONFIG` | 空 | `.env.example` 显式设置为 `configs/view_relative_v1.json` |

当前模板配置：

```text
VIEW_ACTION_SPACE=relative_discrete_v1
VIEW_ACTION_SPACE_CONFIG=configs/view_relative_v1.json
```

V2 配置为：

```text
VIEW_ACTION_SPACE=pose_grid_v2
VIEW_ACTION_SPACE_CONFIG=configs/view_pose_grid_v2.json
```

如果 config path 为空，V1 使用代码内默认参数；V2 没有 pose 列表时无法构造有效 action space。所有相对路径均相对启动命令的工作目录。

## Pipeline 限制与输出

| 变量 | 默认值 | 校验/语义 |
|---|---:|---|
| `MAX_RENDER_RETRIES` | `3` | `0..3`；首次检查之后允许的修复次数，因此默认最多检查 4 个 candidate |
| `MAX_VISUAL_REVISIONS` | `2` | `>= 0`；Verifier round 数最多为该值加 1 |
| `RUNS_DIR` | `data/runs` | Codegen/Pipeline 输出根目录 |

## 覆盖优先级

- 进程环境变量优先于 `.env`，因为 dotenv 使用 `override=False`。
- `run_pipeline --max-visual-revisions` 覆盖 `MAX_VISUAL_REVISIONS`。
- `run_pipeline`/`run_verifier` 的 `--action-space` 与 `--action-space-config` 覆盖对应环境变量。
- `run_verifier --budget` 覆盖 `VERIFIER_ACTION_BUDGET`。
- `--run-id` 只决定 `RUNS_DIR` 下的子目录，不覆盖 `RUNS_DIR`。

`PLAYWRIGHT_CHROMIUM_EXECUTABLE` 由 Pipeline/Verifier 的浏览器解析逻辑读取，但不属于 `Settings` dataclass。可用 `--chromium` 的入口应优先使用显式 CLI 参数。
