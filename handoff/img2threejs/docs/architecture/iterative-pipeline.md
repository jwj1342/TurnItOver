# Iterative Pipeline architecture

完整 Pipeline 将 CodeAgent、确定性 render gate 和 active-view Verifier 连接成有限预算闭环。

## 控制流

```text
reference RGB
    ↓
CodeAgent generate
    ↓
deterministic render check
    ├─ fail → CodeAgent runtime repair → render check
    └─ pass
         ↓
fresh VerifierEpisode
         ↓
verdict + feedback + selected evidence views
         ├─ accept → stop
         └─ revise → CodeAgent visual revise → render check → fresh VerifierEpisode
```

`PipelineController` 负责生成、重试、round、预算和停止。它不解析模型 reasoning 来做执行决策；控制流只依赖结构化 render success、verdict 和预算。

## Render gate

每次生成或修改后的 candidate 必须先通过不依赖 VLM 的 render check：页面加载成功、存在可见 canvas、canvas 非空，并且没有 load、page 或 console error。

失败结果写入 `render_attempts/attempt_XX/check.json`，并作为结构化反馈交给 `CodeAgent.repair_runtime()`。

`MAX_RENDER_RETRIES` 表示首次检查之后的修复次数，范围 `0..3`。默认 3，因此每轮最多检查四个 candidate。耗尽后 Pipeline 以 `render_failed` 结束，不调用该轮 Verifier。

## 两类修改

- **Runtime repair**：candidate 无法可靠渲染，由 render error 驱动，不调用 Verifier。
- **Visual revision**：candidate 已通过 render gate，但 Verifier 返回 `revise`，由视觉反馈和 selected evidence 驱动。

visual revision 后的 HTML 必须重新通过 render gate。

## 中性评估环境契约

CodeAgent prompt 要求 candidate 从初始版本起保持固定、克制的中性评估环境：低强度基础填充、前上方主光、较弱反向补光、后上方轮廓光，以及 `RoomEnvironment + PMREMGenerator` 提供的 `scene.environment`。只设置 `envMapIntensity` 不构成环境反射，只使用强 `AmbientLight` 也会削弱厚度、倒角和遮挡线索。

当前固定基线为：sRGB 输出、ACES Filmic tone mapping、曝光 `1.0`、`RoomEnvironment` 的 PMREM 环境；半球光强度 `0.6`；主光强度 `2.5`、位置 `(5, 8, 6)`；补光强度 `0.8`、位置 `(-4, 3, -5)`；轮廓光强度 `1.0`、位置 `(0, 6, -6)`；背景色 `0xf2f2f2`。这些数值用于跨 revision 保持观察条件稳定，不代表真实场景照明。

该契约目前由 CodeAgent prompt 约束，不是 BrowserSession 对任意 HTML 的强制注入。Runtime repair 和 visual revision 应保留它，不得通过改变曝光或针对当前相机重新布光掩盖问题。

Verifier 必须先在结构视图检查部件、比例、连接、厚度和方向，再检查颜色、金属反光与透明度等外观。它只能报告截图支持的外观症状；仅凭截图不能把近黑外观直接归因为材质参数，因为环境反射、灯光、曝光和法线也可能产生相同现象。

## Fresh verifier invariant

每次代码修改并重新渲染后，都创建新的 `ObservationRuntime` 和 `VerifierEpisode`。上一轮 camera state、observation 集合和截图不会作为下一轮 Verifier 的初始状态。

上一轮 selected evidence 只提供给 CodeAgent 做 revision。若 `revise` 未选出有效 view，controller 会回退使用该轮初始截图。

## Revision budget

`MAX_VISUAL_REVISIONS` 是允许 CodeAgent 执行 visual revision 的次数，不是 Verifier round 数：

| 值 | 最大 Verifier rounds |
|---:|---:|
| 0 | 1 |
| 1 | 2 |
| 2 | 3 |

最后一次 revision 后仍会运行 fresh Verifier。若其 verdict 仍为 `revise`，状态为 `max_visual_revisions`，`accepted=false`。

## 状态与退出码

| `status` | `accepted` | Pipeline CLI 退出码 | 语义 |
|---|---:|---:|---|
| `accepted` | `true` | 0 | Verifier 返回 accept |
| `max_visual_revisions` | `false` | 0 | 修改预算耗尽，仍需修改 |
| `render_failed` | `false` | 1 | render repair 预算耗尽 |

自动化必须读取 `summary.json`，不能只看退出码。

## Camera backend 解耦

Controller 持有统一 `CameraActionSpace`，每个 fresh verifier round 创建匹配的 `ObservationRuntime`。Controller 不直接处理 mouse drag、pose 设置或 harness 调用。

V2 runtime 可通过 CLI/环境配置选择，但 candidate 必须满足 harness 契约；普通 CodeAgent 输出当前尚不保证这一点。

## 输出结构

完整规范见 [../data-and-artifacts.md](../data-and-artifacts.md)。关键产物为：

```text
data/runs/<run-id>/
├── reference.<ext>
├── round_00/
│   ├── workspace_generate/
│   ├── code_agent_generate/
│   ├── render_attempts/attempt_00/check.json
│   ├── runtime_repairs/                 # 可选
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

`reference.<ext>` 保留原图扩展名。`final.html` 只表示最后 candidate，不等于 accepted。

## 运行

```bash
python -m scripts.run_pipeline \
  data/assets/scissors_demo/reference.jpg \
  --run-id pipeline-<new-unique-id>
```

临时覆盖 visual revision 次数：

```bash
python -m scripts.run_pipeline \
  data/assets/scissors_demo/reference.jpg \
  --run-id pipeline-<new-unique-id> \
  --max-visual-revisions 2
```

run 目录必须尚不存在。默认 headless；`--headed` 要求可用显示环境。

## Viewer 集成状态

`app.py` 通过独立的 Pipeline parser 读取 `trajectory.json`、`summary.json` 与逐轮目录。版本化 run 可放在 `data/sessions/agent_demo/<run-id>/`；本地 `data/runs/<run-id>/` 也可直接回放。旧路径只作为 trajectory 元数据，Viewer 会按当前 run 目录重新定位截图和 candidate。

相关静态对应测试为 `tests/test_pipeline_controller.py`、`tests/test_render_check.py`、`tests/test_code_agent.py`、Verifier 测试与 `tests/test_config.py`。本文档修改不运行实验或测试。
