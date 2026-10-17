# Active-view Verifier architecture

VerifierAgent 是与 CodeAgent 分离的 OpenHands Agent。它不共享 CodeAgent conversation，也不拥有 FileEditorTool 或 TerminalTool；它只能通过受限观察工具检查当前 Three.js render，并输出结构化 verdict、反馈和证据视图。

## 数据流

```text
reference RGB + candidate HTML
            ↓
      VerifierAgent
            ↓
      VerifierEpisode
            ↓
 CameraActionSpace + ObservationRuntime
            ↓
       BrowserSession
            ↓
 screenshot returned to the same verifier conversation
```

高层 Pipeline 只依赖 `VerifierEpisode` 和结构化结果，不依赖具体鼠标动作或 pose harness。

## 结构化结果

Verifier 最终提交：

```text
verdict = accept | revise
feedback
selected_view_ids
observations
exhausted_budget
action_space
```

- `accept`：在当前 observation budget 的证据下，没有重要且可操作的 mismatch。
- `revise`：仍存在需要 CodeAgent 修改的视觉问题。

`accept` 是工程停止信号，不应直接解释为经过校准的研究质量标签。

## CameraAction 抽象

统一动作包含 `type`、`name` 和 `params`。每个 observation 同时记录 Agent 请求的 `action_payload` 与 runtime 返回的 `runtime_metadata`，以区分动作意图和浏览器实际状态。

## V1：`relative_discrete_v1`

V1 是当前默认动作空间：

```text
orbit_left / orbit_right
orbit_up / orbit_down
zoom_in / zoom_out
capture
```

`finish` 是 episode 控制动作，不属于 camera runtime。初始截图和 `finish` 不占 observation budget。

默认参数位于 `configs/view_relative_v1.json`。V1 通过 pointer drag / wheel 执行，结果受页面 OrbitControls、rotate speed、damping、camera distance、target 和鼠标映射影响。因此 `orbit_right` 只表示相对离散观察步，不表示精确增加某个 azimuth。

V1 适合工程闭环和 qualitative demo，不适合作为跨 HTML 精确比较 active-view policy 的最终定义。

## V2：`pose_grid_v2`

V2 让 Verifier 从有限、显式的 pose grid 中选择：

```json
{
  "type": "goto_pose",
  "name": "goto_pose",
  "params": {
    "view_id": "az090_el030_d100",
    "azimuth_deg": 90,
    "elevation_deg": 30,
    "distance_scale": 1.0
  }
}
```

默认 pose grid 位于 `configs/view_pose_grid_v2.json`。页面必须暴露同步 camera harness：

```javascript
window.__img2threejsCameraV2 = {
  setPose({ azimuth_deg, elevation_deg, distance_scale }) { /* render */ },
  getPose() { return { azimuth_deg, elevation_deg, distance_scale }; }
};
```

缺少 harness 时，V2 明确失败：

```text
pose_grid_v2 requires deterministic camera harness
```

禁止 silent fallback 到 V1。接口测试 fixture 位于 `tests/fixtures/pose_harness_v2.html`。

## Canonical object frame 限制

V2 runtime、action-space 和 harness 测试已经实现，但普通 CodeAgent 生成的 HTML 尚未被强制实现完整 canonicalization。正式跨样本实验还需统一 object center、scale、front direction、world up、canonical framing distance 和 camera FOV。

在这些定义固定之前，不能仅因使用 V2 配置就声称跨 candidate 的 pose 完全可比。

## Observation budget

`VERIFIER_ACTION_BUDGET` 只计算主动观察动作。初始 screenshot 免费，`finish` 免费；预算耗尽后仍可提交最终反馈。

`VERIFIER_AGENT_MAX_STEPS` 独立限制 OpenHands reasoning/tool turn。两个预算不可合并。

若 OpenHands conversation 首次结束但没有调用 `finish`，VerifierAgent 会在同一 conversation 中追加一次只能提交 `finish` 的收尾消息，并最多再运行一次。该恢复不会增加 observation budget，也不会伪造 verdict。第二次仍未 `finish` 时任务明确失败，并在 `verifier/incomplete_agent.json` 保存已有 observations、events、reasoning、剩余预算和终止信息。

## 输出与独立运行

```text
<out>/verifier/
├── view_00_initial.png
├── view_01_<action-or-pose>.png
├── ...
└── trace.json

<out>/verifier_agent.json
```

V1：

```bash
python -m scripts.run_verifier reference.png candidate.html \
  --out data/runs/verifier-demo \
  --action-space relative_discrete_v1 \
  --action-space-config configs/view_relative_v1.json
```

V2：

```bash
python -m scripts.run_verifier reference.png candidate-with-harness.html \
  --out data/runs/verifier-v2-demo \
  --action-space pose_grid_v2 \
  --action-space-config configs/view_pose_grid_v2.json
```

第二条命令只适用于已实现 V2 harness 的 candidate，不是普通 CodeAgent 输出的通用入口。

相关静态对应测试为 `tests/test_verifier_episode.py`、`tests/test_verifier_agent.py`、`tests/test_verifier_action_space.py` 和 `tests/test_config.py`。本文档修改流程不运行这些测试。
