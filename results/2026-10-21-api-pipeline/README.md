# 真实 API 闭环运行包

本目录的 `summary.json` 是全部原始台账（每个仅一次）的离线汇总，`environment.json` 保存 Python 包、Node/Chromium、源码及真实 runtime 哈希。`artifacts.tar.gz` 保存2026-10-20–21的真实运行与全部排错/失败历史；它包含私有评分参考与标签，**不得整体作为模型输入**。模型实际收到的输入以各 `*-api/api_calls/*/request.json` 为准。

包含的实验：`output/api-prepared-v2` 共享真实 API 初始程序；`output/api-minimal-v6` 为门轴案例的限定分段恢复（原失败 `v5` 完整保留）；`output/api-paired-v6` 为其余两个案例的完整流式配对。其他 v1–v4 为排错/失败记录，不计效果，全部计入成本。没有把脚本 API 测试 fixture 打包为实验。

## 解包和核验

从这个提交的仓库根目录执行；选用没有同名 `output/api-*` 的新 checkout，避免覆盖旧运行：

```bash
(cd results/2026-10-21-api-pipeline && sha256sum -c archive.sha256)
tar -xzf results/2026-10-21-api-pipeline/artifacts.tar.gz
sha256sum -c results/2026-10-21-api-pipeline/artifact-files.sha256
```

归档只包含相对路径的普通文件。`artifact-files.sha256` 枚举归档中的每个文件；真实PNG位于各 verifier 的 `observations/`，HTML画廊为 `index.html`；每次调用的 `images.json` 及 API `request.json` 绑定实际输入图像。SSE 的 `response.sse` 是真实接收字节，`response.json` 是按协议组装的响应，不是一份原始非流式响应。首次400发生在错误正文记录功能添加前，原目录没有该正文；第二次诊断有脱敏错误字段。

## 不调用 API 的评分复现

准备 `pyproject.toml` 所列 Python 依赖和 web 的 npm 依赖（实际版本见 environment）；Playwright 要有兼容 Chromium。归档含本次使用的 `web/dist`；解包后不要重建，否则源码/依赖差异可能触发 runtime 哈希保护。当前机器可用：

```bash
export PYTHONPATH="$PWD"
export PLAYWRIGHT_BROWSERS_PATH="$PWD/../TurnItOver/.cache/ms-playwright"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PY=../TurnItOver/.venv-local/bin/python
# 本命令会重新写入 scores.private.json；先复制原评分保留对比。
cp output/api-minimal-v6/scores.private.json output/api-minimal-v6/scores.original.json
cp output/api-paired-v6/scores.private.json output/api-paired-v6/scores.original.json
$PY scripts/run_api_pipeline.py score --prepared output/api-prepared-v2 --out output/api-minimal-v6
$PY scripts/run_api_pipeline.py score --prepared output/api-prepared-v2 --out output/api-paired-v6
```

评分只使用本地 Chromium，不需要 `.env` 或模型 API。逐状态几何/结论应相同；运行时间不保证逐字节相同。c02 的来源台账采用相对路径及SHA绑定，解包时必须同时保留 `api-minimal-v5-api` 和 `api-minimal-v6-api`。

重新汇总所有调用（输出必须为新文件）：

```bash
$PY scripts/summarize_api_pipeline.py \
  --run output/api-minimal-v6 --run output/api-paired-v6 \
  --ledger output/api-prepared-v1-api/ledger.json \
  --ledger output/api-diagnostic-v1-api/ledger.json \
  --ledger output/api-prepared-v2-api/ledger.json \
  --ledger output/api-minimal-v2-api/ledger.json \
  --ledger output/api-minimal-v3-api/ledger.json \
  --ledger output/api-minimal-v4-api/ledger.json \
  --ledger output/api-minimal-v5-api/ledger.json \
  --ledger output/api-minimal-v6-api/ledger.json \
  --ledger output/api-paired-v6-api/ledger.json --out output/api-recomputed-summary.json
```

全新付费运行及自己的公开参考输入见 [运行说明](../../docs/api-pipeline.md)。模型配置及返回ID可冻结核验，供应商权重、采样与路由未被外部证明；本包提供原始证据重放，不保证新API调用复现同一回答。货币费用未知，4次调用缺usage。详细效果、边界与历史测试失败见 [实验报告](../../docs/api-pipeline-report.md)；审查见 [完整性审查](../../docs/api-pipeline-audit.md)。
