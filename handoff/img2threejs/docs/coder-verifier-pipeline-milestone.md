# Milestone 1–5 阶段总结

本文保留 M1–M5 的阶段演进视角。当前稳定使用说明以 [文档索引](index.md)、[Active-view Verifier 架构](architecture/active-view-verifier.md) 和 [Iterative Pipeline 架构](architecture/iterative-pipeline.md) 为准。所有脚本命令均从仓库根目录以模块方式运行。

## Milestone 1 — Browser Runtime

**主题**
建立可执行模型生成 Three.js HTML 的浏览器运行环境，并支持基础主动视角操作。

**结构**

```text
Three.js HTML
    ↓
Playwright / Chromium
    ↓
BrowserSession
    ↓
orbit / zoom / capture
```

**主要输出**

```text
data/runs/runtime-smoke/
├── render.png
├── 01_orbit_right.png
├── 02_zoom_in.png
├── 03_orbit_up.png
└── result.json
```

`render.png` 是初始 render-check 截图；动作截图从 `01_` 开始编号。`result.json` 记录 load、render check、动作结果以及 page/console errors。

**运行**

```bash
python -m scripts.runtime_smoke generated.html \
  --out data/runs/runtime-smoke \
  --actions orbit_right zoom_in orbit_up
```

`--out` 允许指向已有目录，可能保留或覆盖同名文件；需要独立证据时应使用新的输出目录。

---

## Milestone 2 — Render Validity & Retry

**主题**
判断生成代码是否真正可渲染，并为后续 CodeAgent 提供确定性的 runtime 错误反馈。

**结构**

```text
candidate.html
    ↓
BrowserSession
    ↓
Render Check
    ├─ PASS
    └─ FAIL → structured error → retry
```

检查要求页面成功加载、存在可见 canvas、canvas 像素不是空白或近似常量，并且没有 load、page 或 console error。

**主要输出**

```text
render_attempts/attempt_XX/
├── candidate.html
├── render.png        # 找到 canvas 时出现
└── check.json        # 持久化的 RenderCheckResult
```

`MAX_RENDER_RETRIES` 表示首次检查之后允许的修复次数，范围为 `0..3`。默认值 3 表示每轮最多检查四个 candidate。单独的 `runtime_smoke` 只检查一次；有界 retry 由 Codegen/Pipeline 调用。

**运行**

```bash
pytest tests/test_render_check.py tests/test_render_errors.py -v
```

---

## Milestone 3 — OpenHands CodeAgent

**主题**
让独立的 Qwen + OpenHands CodeAgent 从 RGB 图像生成 Three.js，并在 runtime 失败时自动修复代码。

**结构**

```text
Reference RGB
    ↓
OpenHands CodeAgent
    ↓
candidate.html
    ↓
Render Check
    ├─ PASS
    └─ FAIL → CodeAgent runtime repair
```

CodeAgent 拥有：

```text
FileEditorTool
TerminalTool
```

**主要输出**

```text
candidate / final HTML
CodeAgent visible reasoning
OpenHands events
render attempts
trajectory.json
summary.json
```

**运行**

```bash
python -m scripts.run_codegen \
  data/assets/scissors_demo/reference.jpg \
  --run-id m3-<new-unique-id>
```

Codegen 在 `RUNS_DIR/<run-id>/` 创建新目录；该目录必须尚不存在，不会续跑或覆盖。

---

## Milestone 4 — Active VerifierAgent

**主题**
增加第二个独立 OpenHands Agent，让 Verifier 主动改变观察视角并给 CodeAgent 提供视觉反馈。

**结构**

```text
Reference RGB + Current Render
          ↓
     VerifierAgent
          ↓
   CameraActionSpace
          ↓
 ObservationRuntime
          ↓
 new screenshot
          ↓
 same verifier conversation
```

当前默认使用 **V1：离散相对动作**：

```text
orbit_left / right
orbit_up / down
zoom_in / out
capture
```

同时已实现受限的 **V2：确定性 pose grid runtime**：

```text
goto_pose(view_id)
```

V2 candidate 必须实现 `window.__img2threejsCameraV2.setPose/getPose` harness。普通 CodeAgent 输出尚不保证满足该契约；缺少 harness 时 V2 明确失败，不会回退到 V1。跨样本正式比较还需要统一 canonical object frame。

Verifier 最终输出：

```text
verdict = revise | accept
feedback
selected_view_ids
```

**主要输出**

```text
view_00_initial.png
view_01_*.png
...
trace.json
verifier_agent.json
```

**运行**

```bash
python -m scripts.run_verifier reference.png candidate.html \
  --out data/runs/verifier-demo
```

正常得到 `accept` 或 `revise` 都返回退出码 0；调用方应读取 `verifier_agent.json.verdict`。完整 V1/V2 约束见 [Active-view Verifier 架构](architecture/active-view-verifier.md)。

---

## Milestone 5 — Full Iterative Pipeline

**主题**
将 CodeAgent、Render Check 和 VerifierAgent 接成完整自动迭代闭环。

**结构**

```text
Reference RGB
    ↓
CodeAgent generate
    ↓
Render Check
    ├─ FAIL → runtime repair
    └─ PASS
         ↓
   Fresh VerifierEpisode
         ↓
  verdict + feedback
    ├─ accept → STOP
    └─ revise
         ↓
 CodeAgent visual revise
         ↓
      new render
         ↓
 Fresh VerifierEpisode
         ↓
        ...
```

关键约束：

> 每次代码修改后都重新创建 VerifierEpisode，不复用上一轮观察结果。

`MAX_VISUAL_REVISIONS=2` 表示最多修改两次，因此最多有三轮 Verifier 检查。

**主要输出**

```text
data/runs/<run-id>/
├── reference.<ext>
├── round_00/
│   ├── workspace_generate/
│   ├── code_agent_generate/
│   ├── render_attempts/attempt_00/check.json
│   ├── runtime_repairs/              # 仅发生 runtime repair 时出现
│   ├── verifier/trace.json
│   └── verifier_agent.json
├── round_01/
│   ├── workspace_visual_revise/
│   ├── code_agent_visual_revise/
│   └── ...
├── final.html
├── trajectory.json
└── summary.json
```

`reference.<ext>` 保留输入扩展名；`final.html` 只是最后一个 candidate，不表示已被 Verifier 接受。

| 最终状态 | `accepted` | CLI 退出码 | 含义 |
|---|---:|---:|---|
| `accepted` | `true` | 0 | Verifier 返回 `accept` |
| `max_visual_revisions` | `false` | 0 | 修改预算耗尽，最后一轮仍为 `revise` |
| `render_failed` | `false` | 1 | render repair 预算耗尽 |

因此自动化必须读取 `summary.json.status` 和 `summary.json.accepted`，不能只凭退出码判断成功。

**运行**

```bash
python -m scripts.run_pipeline \
  data/assets/scissors_demo/reference.jpg \
  --run-id m5-<new-unique-id>
```

Pipeline run 目录必须尚不存在。默认使用 headless Chromium；`--headed` 需要可用显示环境。完整控制流和输出契约见 [Iterative Pipeline 架构](architecture/iterative-pipeline.md)。

---

## 整体演进

```text
M1  浏览器能运行、能转相机
 ↓
M2  能判断代码是否真正可渲染
 ↓
M3  CodeAgent 能生成和修复 Three.js
 ↓
M4  VerifierAgent 能主动找视角并给反馈
 ↓
M5  CodeAgent ↔ VerifierAgent 自动闭环
 ↓
M6  Streamlit Viewer 回放完整 Pipeline trajectory
```

当前 Streamlit Viewer 同时支持 `data/sessions/human_demo/` 下的 Chatbox session、`data/sessions/agent_demo/` 下的版本化 Pipeline run，以及本地 `data/runs/<run-id>/trajectory.json`。
