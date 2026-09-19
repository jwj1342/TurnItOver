# Getting started

所有命令都从仓库根目录运行。配置文件路径、输入路径和默认输出路径均相对当前工作目录解析。

## 1. Chatbox Viewer 环境

Viewer 使用 `requirements.txt`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows 激活命令：

```powershell
.venv\Scripts\activate
```

Viewer 支持两类数据：Chatbox session 放在 `data/sessions/human_demo/<session-id>/session.json + resources/`；版本化 Pipeline run 放在 `data/sessions/agent_demo/<run-id>/`。本地 `data/runs/<run-id>/` 也会被直接发现。旧 Codex JSONL 不受支持。

## 2. Agent/Pipeline 环境

`requirements-agent.txt` 会包含基础依赖以及 OpenHands SDK/Tools。当前 OpenHands 上游版本要求 Python 3.12+；此要求只针对 Agent 工作流，不在本文档中扩展为 Viewer 的硬性版本声明。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-agent.txt
playwright install chromium
cp .env.example .env
```

填写 `.env` 中的模型配置：

```text
QWEN_BASE_URL=https://your-endpoint/v1
QWEN_API_KEY=...
QWEN_MODEL=openai/your-qwen-model
```

自定义 OpenAI-compatible endpoint 通常需要 `openai/` provider 前缀，具体取决于 OpenHands/LiteLLM 对服务端的识别方式。

## 3. 运行完整 Pipeline

仓库内已有参考图：

```text
data/assets/scissors_demo/reference.jpg
```

选择一个尚不存在的 run ID：

```bash
python -m scripts.run_pipeline \
  data/assets/scissors_demo/reference.jpg \
  --run-id pipeline-<new-unique-id>
```

Pipeline/codegen 会以 `exist_ok=False` 创建 run 目录；重复使用已有 ID 会失败，不会续跑或覆盖。

## 4. 浏览器运行条件

- 默认 headless，不需要图形桌面。
- `--headed` 需要可用的显示环境；无 `$DISPLAY` 的服务器不应直接使用。
- `run_pipeline` 的 `--chromium` 优先于 `PLAYWRIGHT_CHROMIUM_EXECUTABLE`；未显式指定时再尝试环境变量和系统 Chromium/Chrome。
- `run_verifier` 没有 `--chromium` 参数，会依次尝试 `PLAYWRIGHT_CHROMIUM_EXECUTABLE`、系统 Chromium/Chrome 和 Playwright 环境。
- `run_codegen --chromium` 和 `runtime_smoke --chromium` 可显式指定浏览器路径。
- candidate HTML 若依赖外部 CDN，运行结果仍受网络可达性影响。

## 5. 判断运行结果

运行结束后检查 `data/runs/<run-id>/summary.json`：

```json
{
  "status": "accepted | max_visual_revisions | render_failed",
  "accepted": true
}
```

`max_visual_revisions` 的进程退出码仍可能为 0，因此自动化必须读取结构化结果，不能把退出码 0 等同于 Verifier 接受。

配置细节见 [configuration.md](configuration.md)，输出结构见 [data-and-artifacts.md](data-and-artifacts.md)。

如果需要按项目演进顺序理解各模块，参见 [Milestone 1–5 阶段总结](coder-verifier-pipeline-milestone.md)；当前稳定接口仍以 CLI、配置和专题架构文档为准。
