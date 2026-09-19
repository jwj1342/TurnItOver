# Image-to-Three.js Pipeline + Trajectory Viewer

本仓库同时包含两条互补工作流：

- **实验 Pipeline**：由 OpenHands/Qwen CodeAgent 生成 Three.js HTML，经确定性渲染检查后，由主动视觉 Verifier 检查并驱动有限轮修改。
- **轨迹 Viewer**：用 Streamlit 查看 Chatbox 导出会话或原生 Pipeline run 的参考图、渲染图、reasoning、Verifier 证据、HTML 代码和 checkpoint 差异。

当前 Viewer 将可版本化会话统一放在 `data/sessions/` 下，并按来源分为 `human_demo/` 与 `agent_demo/`。另外仍支持直接查看本地 `data/runs/<run-id>/` Pipeline 输出。旧 Codex JSONL 仍不在支持范围内。

## 当前能力

| 能力 | 状态 | 说明 |
|---|---|---|
| Chatbox trajectory Viewer | 已实现 | 入口为 `streamlit run app.py` |
| Three.js runtime smoke check | 已实现 | 检查页面、canvas、非空画面和浏览器错误 |
| OpenHands/Qwen CodeAgent | 已实现 | 生成 HTML，并可根据结构化渲染错误修复 |
| Active-view Verifier | 已实现 | 独立 conversation、受限相机观察动作、结构化 verdict |
| V1 `relative_discrete_v1` | 当前默认 | 使用 pointer drag / wheel，不承诺精确相机角度 |
| V2 `pose_grid_v2` runtime | 已实现但受限 | candidate 必须实现确定性 camera harness；普通 CodeAgent 输出尚不保证满足该契约 |
| CodeAgent ↔ Verifier Pipeline | 已实现 | 支持 render repair、visual revision 和有限预算停止 |
| Pipeline native trajectory Viewer | 已实现 | 直接读取 `data/sessions/agent_demo/<run-id>/` 或本地 `data/runs/<run-id>/` |

完整文档入口见 [docs/index.md](docs/index.md)。

## 环境

本项目已有 Conda 环境 `img2threejs`，进入仓库后可直接激活：

```bash
conda activate img2threejs
```

下面各工作流也可以使用项目目录中的 Python `venv`，二者任选其一，不需要同时激活。

## Quick start：Trajectory Viewer

Viewer 使用基础依赖，不要求安装 OpenHands Agent SDK：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows 激活命令为：

```powershell
.venv\Scripts\activate
```

可版本化 demo 统一使用：

```text
data/sessions/
├── human_demo/
│   └── <session-id>/
│       ├── session.json
│       └── resources/
│           ├── input.png
│           └── resource-*.png
└── agent_demo/
    └── <run-id>/
        ├── trajectory.json
        ├── summary.json
        ├── reference.png
        ├── final.html
        └── round_*/
```

`human_demo` 对应人工/Chatbox 轨迹，`agent_demo` 对应 CodeAgent ↔ Verifier Pipeline 轨迹。两个 discovery parser 都支持在各自根目录下继续增加分组层级，只要最终目录分别包含 `session.json` 或 `trajectory.json` 即可。

历史 Chatbox 数据中的 `inpuy.jpg` 仍会被兼容识别；新数据应统一使用 `input.*`。所有版本化 session 都可能包含 prompt、reasoning、图片与模型元数据，提交前必须检查隐私内容。

## Quick start：实验 Pipeline

Agent 工作流使用 `requirements-agent.txt`，当前 OpenHands SDK/Tools 上游要求 Python 3.12+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-agent.txt
playwright install chromium
cp .env.example .env
```

在 `.env` 中至少填写：

```text
QWEN_BASE_URL=https://your-endpoint/v1
QWEN_API_KEY=...
QWEN_MODEL=openai/your-qwen-model
```

从仓库根目录运行，并为每次 Pipeline/Codegen 选择一个尚不存在的 run ID：

```bash
python -m scripts.run_pipeline \
  data/assets/scissors_demo/reference.jpg \
  --run-id pipeline-<new-unique-id>
```

默认使用 headless Chromium。`--headed` 只适用于已有图形显示环境；无 `$DISPLAY` 的服务器应保持 headless，或由使用者自行配置 Xvfb。

Pipeline 输出默认位于 `data/runs/<run-id>/`。运行 `streamlit run app.py` 后，这些本地 run 会直接出现在 Session 下拉框中，无需转换。若某个 run 需要随仓库版本化作为固定 demo，可将完整目录复制到 `data/sessions/agent_demo/<run-id>/`。

是否被 Verifier 接受必须读取：

```text
summary.json.status
summary.json.accepted
```

不能只依据进程退出码判断成功：`max_visual_revisions` 会正常结束进程，但 `accepted` 仍为 `false`。

## 其他入口

以下命令都应从仓库根目录以模块方式运行：

```bash
# 对已有完整 HTML 做确定性渲染检查和相机动作 smoke test
python -m scripts.runtime_smoke path/to/generated.html \
  --out data/runs/runtime-smoke \
  --actions orbit_right zoom_in orbit_up

# 只运行 CodeAgent + render retry
python -m scripts.run_codegen path/to/reference.png --run-id codegen-<new-unique-id>

# 对已有 candidate 运行独立 Verifier
python -m scripts.run_verifier reference.png candidate.html \
  --out data/runs/verifier-demo
```

参数、退出码和覆盖规则见 [CLI reference](docs/cli-reference.md)。

## 数据与输出概览

```text
data/
├── sessions/
│   ├── human_demo/      # 可版本化 Chatbox / 人工轨迹
│   └── agent_demo/      # 可版本化 Pipeline / Agent 轨迹
├── assets/              # 实验参考图等受控输入资产
├── runs/                # 本地 Pipeline/Codegen/Verifier/runtime 产物；被 gitignore
└── bundles/             # 可选的 Viewer 分片数据包
```

完整的数据契约、输出目录和隐私说明见 [data-and-artifacts.md](docs/data-and-artifacts.md)。

## Viewer checkpoint 语义

### Human / Chatbox

Chatbox parser 读取 `session.json` 顶层 `messages`；`messageForksHash` 不会混入主 trajectory。一个 checkpoint 对应一次 assistant 输出完整 HTML 的完成状态：

```text
user prompt
    ↓
assistant visible reasoning
    ↓
assistant complete HTML
    ↓
checkpoint
```

首个带图像的 user message 对应真实 RGB reference；后续图像按时间顺序映射为已有 checkpoint 的 render。与 reference 内容完全相同的 `resource-*` 副本会按哈希过滤。界面只展示 Chatbox 实际保存的 visible reasoning，不尝试恢复隐藏推理。

### Agent / Pipeline

Pipeline parser 将每个 `round_XX` 映射为一个 checkpoint：

```text
CodeAgent generate / visual_revise
    ↓
render gate (+ optional runtime repair)
    ↓
Verifier active views + verdict + feedback
    ↓
checkpoint
```

Viewer 以最后一个成功的 `render_attempts/attempt_XX/candidate.html` 作为该 round 的 authoritative HTML，而不是无条件使用最初的 `code_agent_*/code.html`。这样在发生 runtime repair 时，代码 Diff、Live Render 与 Verifier 实际检查的候选保持一致。

`trajectory.json` 中记录的旧 `data/runs/...` 路径只作为元数据；Viewer 会基于当前 run 目录重新定位 render 和 verifier screenshots，因此完整 run 被复制、解压或移动后仍可展示。

## 文档导航

- [Getting started](docs/getting-started.md)
- [CLI reference](docs/cli-reference.md)
- [Configuration](docs/configuration.md)
- [Data and artifacts](docs/data-and-artifacts.md)
- [Milestone 1–5 阶段总结](docs/coder-verifier-pipeline-milestone.md)
- [Active-view Verifier architecture](docs/architecture/active-view-verifier.md)
- [Iterative Pipeline architecture](docs/architecture/iterative-pipeline.md)
- [Experiment documentation policy](docs/experiments/README.md)

## 当前限制

- Viewer 支持 Chatbox `session.json + resources/` 和当前 Pipeline `trajectory.json + round_*/` schema；旧 Codex JSONL 不支持。
- Chatbox 图像映射仍依赖 user message 时间顺序与 `resource-*` 文件编号。
- Pipeline Viewer 依赖每个 round 的本地产物目录仍存在；只有 `trajectory.json` 而没有 `round_*/` 时无法恢复完整 evidence。
- 浏览器渲染可能受外部 CDN、浏览器兼容性和候选 HTML 自身错误影响。
- V1 相机动作不可用于宣称跨 HTML 的精确角度一致性。
- V2 只有在 candidate 实现 canonical camera harness 时才具备确定性语义。

## 部署与隐私

`streamlit run app.py` 可用于本地部署，也可部署到兼容的 Streamlit 托管环境。仓库不把任何外部部署实例、分支或访问权限声明为当前事实；这些信息应由部署负责人单独维护。

在确认所有 human/agent session、图片、prompt、reasoning 和模型元数据均已脱敏前，应保持仓库和部署为私有。
