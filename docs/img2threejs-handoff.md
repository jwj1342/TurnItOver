# Img2Three.js prototype 交接说明

## 定位

`handoff/img2threejs/` 是从 `img2threejs-trajectory-demo` 的 `develop` 分支提取的代码快照，用于一次性交付和架构评审。它保留原型自身的源码、Viewer、配置、prompt、测试、文档和受控参考资产，但没有并入正式 `turnitover` Python 包。

这份快照的价值是展示已经实现的工程闭环：CodeAgent 生成独立 Three.js HTML，经确定性 render gate 后交给 active-view Verifier；Verifier 返回结构化 verdict、反馈和证据，CodeAgent 在有限预算内修改，修改后的 candidate 再经过 render gate 和 fresh Verifier。Streamlit Viewer 可回放 Chatbox session 与 Pipeline trajectory。

它不是 TurnItOver Program ABI、浏览器协议或 verifier schema 的替代实现，也不代表相关架构决策已经完成。

## 交付内容与排除项

快照包含：

- `src/` 中的 agents、pipeline controller、runtime、verifier、trajectory writer、Viewer parser 和 prompts；
- `scripts/` 中四个受支持的模块入口；
- `app.py`、两套 action-space 配置、单元/浏览器测试和设计文档；
- `data/assets/scissors_demo/` 中经过控制的参考资产；
- Viewer 基础依赖与隔离的 OpenHands Agent 依赖。

交付明确排除 `data/sessions/`、`data/runs/`、bundles、历史截图堆、模型事件 dump、`agent.json`、`verifier_agent.json`、`node_modules`、日志、缓存、虚拟环境、`.env` 和密钥。旧 Codex rollout JSONL 入口也不在支持范围内。

## 环境与入口

以下命令都从 prototype 根目录运行，不能从 TurnItOver 仓库根目录直接运行相对路径：

```bash
cd handoff/img2threejs
conda activate img2threejs
```

Viewer 只需要 `requirements.txt`：

```bash
pip install -r requirements.txt
streamlit run app.py
```

Agent/Pipeline 需要 Python 3.12+、`requirements-agent.txt`、Chromium 和本地 `.env`。`.env.example` 只记录变量名；实际凭证不得提交。

```bash
pip install -r requirements-agent.txt
playwright install chromium
cp .env.example .env
python -m scripts.run_pipeline \
  data/assets/scissors_demo/reference.jpg \
  --run-id pipeline-<new-unique-id>
```

另外三个受支持入口是：

```bash
python -m scripts.runtime_smoke HTML [--out DIR] [--actions ACTION ...]
python -m scripts.run_codegen REFERENCE [--run-id ID]
python -m scripts.run_verifier REFERENCE HTML [--out DIR]
```

Pipeline 是否被接受必须读取 `data/runs/<run-id>/summary.json` 的 `status` 与 `accepted`。`max_visual_revisions` 可以以退出码 0 结束但仍为 `accepted=false`。

## Artifact 与隐私边界

本地运行默认写入 `data/runs/`，该目录是可再生 artifact，不进入本次交付。若未来选择版本化某个 trajectory，必须先检查参考图、prompt、visible reasoning、模型标识、usage 和所有截图的隐私与授权，并保持完整 provenance。

`render_check.success=true` 只表示当前页面通过可渲染门槛；`accepted=true` 只是给定 observation budget 下的工程停止信号。两者都不能单独支持视觉质量、物理正确性或数据集级效果结论。

## 与 TurnItOver 的重复及待决策项

prototype 与正式仓库在 verifier、浏览器 runtime、模型 wrapper、轨迹记录和修复编排上存在重叠。后续整合应以 TurnItOver 的 Program ABI、`ObservationSession`、动作协议、模型 client 和实验 provenance 为 source of truth，并逐项决定：

1. 将 prototype controller 的 render-before-verifier、runtime repair/visual revision 分离、fresh-verifier 和有限预算停止语义移植到正式 outer loop；
2. 将 HTML candidate 适配为 TurnItOver TypeScript Program ABI，而不是长期维护第二套浏览器 runtime；
3. 评估 trajectory Viewer 是扩展现有 offline report，还是作为独立工具保留；
4. OpenHands 若继续使用，应保持可选依赖，不能成为 TurnItOver 基础环境的硬依赖。

在这些决策完成前，`handoff/img2threejs/` 只作为可运行、可测试的 prototype 快照维护。
