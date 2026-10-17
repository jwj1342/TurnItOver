# AGENTS.md

本文件适用于整个仓库。若未来某个子目录增加更具体的 `AGENTS.md`，该子目录内以更具体的指令为准。

## 项目定位

本仓库包含两条并行但数据格式不同的工作流：

1. **Image-to-Three.js 实验 Pipeline**：CodeAgent 生成 Three.js HTML，确定性 render gate 检查可渲染性，Active-view Verifier 主动观察并驱动有限轮视觉修改。
2. **Chatbox 轨迹 Viewer**：Streamlit 应用读取 Chatbox `session.json + resources/`，展示参考图、渲染图、visible reasoning、HTML 和 checkpoint diff。

不要把两条工作流的数据接口混为一谈。Viewer 分别通过 Chatbox parser 与 Pipeline parser 读取两套数据：前者使用 `session.json + resources/`，后者使用 `trajectory.json + round_*/`；本地 `data/runs/<run-id>/` 也可直接回放。

## 开始工作前

- 先阅读与任务最相关的文档：
  - `README.md`
  - `docs/index.md`
  - `docs/cli-reference.md`
  - `docs/configuration.md`
  - `docs/data-and-artifacts.md`
  - `docs/architecture/active-view-verifier.md`
  - `docs/architecture/iterative-pipeline.md`
- 先运行 `git status --short --branch`。工作树可能包含用户尚未提交的修改；不得覆盖、回退或顺手整理无关改动。
- 使用 `rg` / `rg --files` 搜索实现与引用。
- 以当前代码、`.env.example`、配置文件和已有 artifact 为事实来源；文档或历史记录与代码冲突时，先指出冲突再处理。
- 不读取、打印或提交 `.env` 中的密钥。

## 环境与安装

- Viewer/runtime 基础依赖：`requirements.txt`。
- Agent 依赖：`requirements-agent.txt`；当前 OpenHands SDK/Tools 上游要求 Python 3.12+。
- 浏览器 runtime 使用 Playwright/Chromium。安装浏览器：

```bash
playwright install chromium
```

- 默认使用 headless Chromium。只有在存在图形显示环境时才使用 `--headed`；无 `$DISPLAY` 的服务器不要直接启用 headed。
- candidate HTML 可能依赖外部 CDN。区分网络/CDN 失败与代码本身的渲染错误。

## 受支持入口

所有脚本都从仓库根目录以模块方式运行。不要在文档、建议或自动化中使用 `python scripts/<name>.py`。

```bash
# Chatbox Viewer
streamlit run app.py

# 已有 HTML 的 render/action smoke check
python -m scripts.runtime_smoke HTML [--out DIR] [--actions ACTION ...]

# CodeAgent + bounded render retry
python -m scripts.run_codegen REFERENCE [--run-id ID]

# 独立 Active-view Verifier
python -m scripts.run_verifier REFERENCE HTML [--out DIR]

# 完整 CodeAgent ↔ Verifier Pipeline
python -m scripts.run_pipeline REFERENCE [--run-id ID]
```

旧 Codex rollout JSONL 不属于受支持输入；不要围绕该 legacy 格式扩展文档或功能。

## 配置事实

- Agent 入口从 `.env` 读取 `Settings`；进程环境变量优先于 `.env`。
- 必填模型配置：`QWEN_BASE_URL`、`QWEN_API_KEY`、`QWEN_MODEL`。
- 主要默认值：
  - `CODE_AGENT_MAX_STEPS=8`
  - `VERIFIER_AGENT_MAX_STEPS=10`
  - `VERIFIER_ACTION_BUDGET=6`
  - `MAX_RENDER_RETRIES=3`
  - `MAX_VISUAL_REVISIONS=2`
  - `VIEW_ACTION_SPACE=relative_discrete_v1`
  - `RUNS_DIR=data/runs`
- `MAX_RENDER_RETRIES` 的语义是“首次检查之后的修复次数”。默认 3 表示每轮最多检查 4 个 candidate。
- `MAX_VISUAL_REVISIONS` 是 CodeAgent 视觉修改次数；最大 Verifier round 数为该值加 1。
- 相对路径按启动命令的工作目录解析，因此命令必须从仓库根目录运行。

## 数据与 artifact 边界

- `data/sessions/`：Chatbox Viewer 输入，当前允许版本化；可能包含 prompt、visible reasoning、模型信息和图片，提交前必须检查隐私。
- `data/assets/`：受控实验输入。文档中的真实示例优先使用 `data/assets/scissors_demo/reference.jpg`。
- `data/runs/`：本地实验产物，默认被 gitignore。目录存在不代表结果已验证或可复现。
- `data/bundles/`：可选的 Viewer 分片数据包。
- 不修改原始 Chatbox `session.json` 来迎合 parser；需要兼容时修改 parser 并添加测试。
- Pipeline/codegen 使用 `exist_ok=False` 创建 run 目录。使用新的 `run-id`，不要覆盖或伪装续跑已有 run。
- Runtime/verifier 的 `--out` 可指向已有目录，可能混入旧文件；正式取证时使用新目录。
- 不删除或重写用户已有 `data/runs/`、session、截图或结果文件，除非用户明确指定目标并授权。

## Pipeline 强制不变量

### Render gate 在 Verifier 之前

每个初始生成或视觉修改后的 candidate 都必须先通过确定性 render check：

- 页面成功加载；
- 存在可见 canvas；
- canvas 不是空白/近似常量画面；
- 没有 load、page 或 console error。

失败结果写入 `render_attempts/attempt_XX/check.json`。失败 candidate 只能进入 bounded runtime repair；在通过 render gate 之前不得调用 Verifier。

最多进行配置允许的修复次数；默认首次检查加 3 次 repair。耗尽后以 `render_failed` 结束，不得把不可渲染 candidate 交给 Verifier。

### Runtime repair 与 visual revision 分离

- Runtime repair 只处理无法可靠渲染的问题，由结构化 render error 驱动。
- Visual revision 只在 render 已通过后执行，由 Verifier feedback 和 selected evidence 驱动。
- Visual revision 产生的新 HTML 必须重新经过 render gate。

### Fresh Verifier

每次代码修改后必须创建新的 `ObservationRuntime` 和 `VerifierEpisode`。不得跨 revision 复用旧 camera state、observation 集合或 verifier conversation。

上一轮 selected evidence 只能作为 CodeAgent revision 输入，不得冒充下一轮 Verifier 的新观察。

### 状态判定

Pipeline 权威状态为：

- `accepted`：`accepted=true`，Verifier 返回 accept。
- `max_visual_revisions`：`accepted=false`，预算耗尽且最后 verdict 仍为 revise。
- `render_failed`：`accepted=false`，render repair 预算耗尽。

`max_visual_revisions` 的 CLI 退出码仍为 0。任何自动化和结果报告都必须读取 `summary.json.status` 与 `summary.json.accepted`，不能只看退出码或 `final.html` 是否存在。

## Camera action space

- V1 `relative_discrete_v1` 是当前默认。`orbit_left/right/up/down` 与 `zoom_in/out` 是相对语义动作，不承诺精确角度，也不适合直接支持跨 HTML 的严格 pose 比较。
- V2 `pose_grid_v2` runtime 已实现，但 candidate 必须提供 `window.__img2threejsCameraV2.setPose/getPose`。
- V2 缺少 harness 时必须明确失败；禁止静默回退 V1。
- 普通 CodeAgent 输出尚未被强制实现完整 canonical object frame。未统一 object center、scale、front、world up、framing distance 与 FOV 前，不得声称跨 candidate pose 完全可比。
- Controller 只依赖 `CameraActionSpace + ObservationRuntime`，不要把 mouse gesture 或 pose backend 分支写回高层控制流。

## Prompt 修改规则

- Prompt 当前使用中文；除非用户明确要求改语言，否则保持中文。
- 物理合理性是独立验收维度，不得降级成一般几何或材质描述。至少覆盖：支撑、连接、关节、厚度、遮挡、穿插和功能关系。
- `src/prompts/code_visual_revise.md` 必须保留 `{feedback}`，因为代码使用 `.format(feedback=feedback)`。
- 修改 prompt 时，先检查调用处、模板变量和现有测试。
- 如果用户只要求审查或方案，先给完整 prompt 草案，不要直接改文件；只有明确授权实现后才落盘。
- Verifier 只能基于可观察证据下结论；区分“不存在”“被遮挡”和“当前视角无法确认”。

## 结果与研究完整性

- `render_check.success=true` 只证明通过当前可渲染门槛，不证明视觉、语义或物理正确。
- 代码中创建了组件，不等于该组件在 render 中可辨认、连接正确或在有效视角可见。
- `accepted=true` 是当前 Verifier 在给定 observation budget 下的工程停止信号，不是经过校准的通用质量标签。
- 单个 run、局部测试或实现存在性不能支持数据集级成功率、质量提升或论文结论。
- 正式结果至少记录：commit、worktree 状态、输入资产、非敏感模型标识、相关配置、action space、预算、run ID、artifact 位置、status、accepted、round/revision 数和已知不确定性。
- `data/runs/pipeline-demo-001` 仅可作为 schema/失败状态示例；其 `accepted=false`，不得描述为成功重建或验证通过。
- 不尝试恢复、推断或声称拥有隐藏 reasoning。只保存和展示模型/OpenHands 实际暴露的 visible reasoning/events。

## 代码修改约定

- 保持现有 Python 风格：类型标注、`pathlib.Path`、小而明确的 dataclass/结构化结果。
- 高层 controller 不解析自由文本 reasoning 来控制执行；决策依赖结构化字段。
- 对错误使用明确异常或结构化失败，不做破坏确定性的 silent fallback。
- 修改 CLI 时同步更新 `docs/cli-reference.md`、`docs/configuration.md` 和相关 README 示例。
- 修改 artifact schema/目录时同步更新 `docs/data-and-artifacts.md` 与实验记录规范。
- 修改 V1/V2 或 Pipeline 不变量时同步更新对应 architecture 文档。
- 不顺带格式化或重写无关文件。

## 验证策略

按改动风险选择最小充分验证。不要把运行真实模型/Pipeline 当成普通单元测试。

### 纯文档改动

- `git diff --check`
- 检查 Markdown 本地链接目标存在。
- 搜索并避免旧式 `python scripts/...` 命令。
- 静态对照 argparse、`.env.example`、`Settings` 和实际输出写入代码。

### 配置、controller 或 render gate

```bash
python -m pytest \
  tests/test_config.py \
  tests/test_pipeline_controller.py \
  tests/test_render_check.py -q
```

### Verifier/action space

```bash
python -m pytest \
  tests/test_verifier_episode.py \
  tests/test_verifier_agent.py \
  tests/test_verifier_action_space.py -q
```

### CodeAgent/trajectory

```bash
python -m pytest \
  tests/test_code_agent.py \
  tests/test_trajectory_writer.py -q
```

浏览器测试可能因 Chromium 不可用而 skip。报告结果时单独说明 passed、failed 和 skipped；不得把全部 skip 写成验证通过。

只有用户明确要求运行真实实验并提供必要凭证/环境时，才启动 CodeAgent、Verifier 或完整 Pipeline。真实运行使用新的 run ID，先说明预计写入位置和外部模型/CDN 依赖。

## 完成与交付

- 结束前再次检查 `git status --short` 和 `git diff --check`。
- 清楚列出修改的文件、删除的文件、实际运行的检查以及未运行的检查。
- 不把“代码已实现”“测试通过”“真实实验有效”混为同一结论。
- 不提交、不 push，除非用户明确要求。用户说“提交当前修改”时只做本地 commit，并只暂存任务相关文件。
