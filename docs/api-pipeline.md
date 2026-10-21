# API 生成—主动验证—反馈修复闭环

分支 `laq/api-pipeline` 基于 `laq/sft-dataset`。本入口只做推理和小规模配对验证，不训练、修改数据集或读取冻结 Test。

## 已有模块与新增接口

| 环节 | 复用代码 | 本次新增 |
| --- | --- | --- |
| 初始 API 生成 | `models.generation.generate_program` | 初始程序只生成一次，两组共享 |
| Chromium 执行/渲染 | `render.session.ObservationSession`、`render.gate` | 配对初始真实截图一致性检查 |
| 观察动作 | `policy.loop.execute_observation` | 不增加第二套动作执行器 |
| 主动/固定验证 | `verifier.runner`、`VisionPolicy`、现有 schema | `exact_budget=True`，两组用满 K 后共用最终协议 |
| 反馈修复 | `repair.iterative.run_iterative`、`propose_repair` | 配对编排、共享冻结 API callable |
| 独立评分 | `gt_validator.validate_code`、`GeometryAlignmentChecker` | 部件/类型评分、逐步动作重放和几何见证检查 |
| API 记录 | 现有 REST adapter | 可选原始请求/响应记录，统一调用台账和返回模型身份检查 |

新增模块位于 `turnitover/api_pipeline/`，按 API 记录、案例、配对编排、评分分文件。已有 `turnitover iterate` 默认行为保持兼容。

## 协议

Generator、Active、Fixed 和修复器使用同一个 `FrozenAPI` 配置：指定 provider/model、token 上限、timeout、temperature=1（k3-256k 接口强制要求），不切换模型、不自动重试传输失败。首次 temperature=0 被拒后的独立诊断确认只允许 1，已按接口要求修正；这不属于性能调参。首次响应的 model 身份冻结，后续不一致立即停止。API 模型 ID 及返回身份不等于供应商权重快照，供应商路由和非确定性仍限制逐次复现。

参考输入为一个自编的三部件、双关节物体的公开设计要求（尺寸、位置、标识、运动规则）和本次 Chromium 新渲染的参考图。它不是 PartNet 新训练数据。独立作者参考实现只在 `private/reference.ts` 中供评分使用；源码、腐蚀标签、评分、私有几何不会进入 Verifier 或修复反馈。公开设计要求本身属于任务输入，不能当作诊断标签。

初始生成及不可执行程序的有限 runtime repair 只执行一次；可渲染的最终初始程序随后冻结。受控案例 c00 为该 API 程序，c01 通过其公开 ABI 给 drawer 加位置偏移，c02 通过公开 ABI 把 door_hinge 改成 X 轴。两组都从同一份受控源码出发，候选源码只供 Generator 修复使用。Verifier 仅收到公开任务、参考图、合法操作标识及自身已获取的证据。

每轮创建全新浏览器 session，采用相同 Chromium、渲染尺寸、视角集合和阈值。两组初始程序和 render gate 图片检查一致；这些 gate 图片不会免费进入 Verifier。Fixed 在观察前写下基于公开合法操作清单的固定计划，默认 stats → front → oblique_fr → 首个关节 limit → state_delta → zero → 后续关节。短计划确定性循环补足 K。Active 每一步只看已有历史选择下一动作；在 K 前返回 verdict 会触发已存在的 schema 校验反馈，每决策最多三次模型尝试。所有尝试记录并计入 API 成本，不能算免费。

两组观察预算都为 K=4，修复上限均为一次视觉修改、每轮一次 runtime repair。最终诊断协议完全相同。Active 选择观察需要额外 API 调用，等观察预算不代表等 API 预算。运行顺序在案例间交错，不声称消除了时间/缓存差异。修复不以 GT 分数停止或回滚；`accepted` 仅表示 Verifier 判 pass。

## 执行

复用兼容 Python/Playwright 环境和对应的 Chromium 缓存；构建 web（`cd web && npm ci && npm run build`）。以下示例从仓库根目录运行，当前机器可使用兄弟 checkout 的虚拟环境：

```bash
export PYTHONPATH="$PWD"
export PLAYWRIGHT_BROWSERS_PATH="$PWD/../TurnItOver/.cache/ms-playwright"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PY=../TurnItOver/.venv-local/bin/python

# .env 中已有 TIO_TEACHER_PROVIDER/MODEL/BASE_URL/API_KEY_ENV；不复制或输出密钥。
# 一次共享生成；默认读取现有 teacher 配置作为所有角色的冻结 API。
$PY scripts/run_api_pipeline.py prepare --out output/api-prepared-new --max-calls 2 --stream

# 最小受控修复案例；先检查日志/图片/评分，再继续小规模配对。
$PY scripts/run_api_pipeline.py run --prepared output/api-prepared-new \
  --out output/api-minimal-new --cases c02 --max-calls 20 --request-timeout 300 --max-output-tokens 8192 --stream
$PY scripts/run_api_pipeline.py score --prepared output/api-prepared-new --out output/api-minimal-new

# 剩余 clean 和位置偏移；共享同一个冻结生成结果。
$PY scripts/run_api_pipeline.py run --prepared output/api-prepared-new \
  --out output/api-paired-new --cases c00 c01 --max-calls 30 --request-timeout 300 --max-output-tokens 8192 --stream
$PY scripts/run_api_pipeline.py score --prepared output/api-prepared-new --out output/api-paired-new
```

所有输出目录必须为新目录。API 记录放在对应 `*-api/`，记录 payload（含真实传输图片）、原始成功响应、限长脱敏错误字段、prompt、图像及哈希、token usage、延迟、返回模型身份、失败状态与 provider 返回的 cost；不保存密钥/请求头/端点地址。调用数限制包括失败尝试，不是美元硬上限。没有返回货币账单时 cost 为 null，不能报告零费用，也不能用其他服务的公开价格冒充本 gateway 账单。

`prepare` 保存共享生成/修复历史、参考 PNG、输入清单及冻结配置。`run` 保存每个 arm 的 `initial.ts`、各轮程序版本、render gate PNG、Verifier 的 `trajectory.jsonl`/`observations`/`feedback.json`/模型记录、Generator 补丁与 `final.ts`。两阶段通过冻结文件哈希和模型身份/采样配置核对输入。共享初始生成已用4096输出上限完成；比较阶段统一用8192（4096曾全部耗于推理而无诊断JSON），两组的每个决策、诊断和修复调用共用同一上限，根清单分别记录共享生成和比较配置。`--request-timeout` 只改变客户端等待时间，记录在各次运行中，不进入模型 payload；原120秒超时的运行保留，新配对两组统一300秒。`score` 完全不调用 API；私有结果写入 `scores.private.json`。

## 评分与边界

独立 GT 检查程序 contract、静止和每关节 mid/limit 的部件几何，以及运行时预算。几何阈值沿用现有 checker，不能衡量全部语义、材质、碰撞或物理正确性。质量同时报告全部检查是否通过及几何/运行时 invariant 通过比例；首次通过轮次和最终通过必须与 Verifier accepted 分开。

成本总账以 `*-api/ledger.json` 为准，包括有 usage 的失败响应；旧迭代器的 `total_tokens` 只汇总成功返回的 ModelResult，不能替代完整账单。

只有独立初始程序检查通过，c00/c01/c02 的精确注入标签才有效，否则类型/定位统计留空。初始缺陷检测、类型集合、部件 TP/FP/FN、关节定位和引用存在性分开记录。证据检查在模型完成后重放同一动作，通过独立参考实现比较该时刻的部件几何，要求找到引用中的图像观察、匹配类型和部件及实际几何不一致。这是“几何见证一致性”必要条件，不能证明遮挡图像中缺陷可见，也不能证明自然语言解释正确；不将它包装成完整语义证据正确率。

受控案例只检验一个程序族、一次采样和有限修复预算。不能据此声称 Active 普遍优于 Fixed，不能将不同观察图像、schema 合法或像素差异直接当作推理能力提升。失败运行完整保留，不自动续跑，不通过改模型/提示寻找更好成绩。

离线总账入口 `scripts/summarize_api_pipeline.py --run <run> --ledger <api/ledger.json> --out <new-summary.json>`，`--run` 和 `--ledger` 可重复提供；应同时提供准备、排错、失败运行和成功实验的全部台账，避免漏算。

## 长推理的流式传输与限定恢复

本地等待时间不能消除边缘代理的超时：真实非流式请求曾在约126秒返回524，与 [Cloudflare 524文档](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/) 所述约125秒默认读取超时相符。`--stream` 采用兼容 Chat Completions SSE，保存实际接收的 `response.sse`，再组装 `response.json`；只有完整终态才能作为模型结果。usage请求遵循 [Chat Completions streaming events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)。流式是否可避免当前网关的超时，仍以实际运行记录为准。300秒是socket等待设置，不是流式请求的总墙钟硬上限。

正常运行可在上述 `prepare` / `run` 命令加 `--stream`。已有真实观察全部获取、只剩最后诊断被524/传输等待中断时，可使用限定恢复入口；其他中断点会拒绝：

```bash
$PY scripts/run_api_pipeline.py recover-final --prepared output/api-prepared-v2 \
  --source-run output/api-minimal-v5 --out output/api-minimal-v6 \
  --max-calls 5 --max-output-tokens 8192 --request-timeout 300 --stream
```

该入口只适用于已完成一次修复、Active的K步重新观察全部存在、Fixed尚未启动的单案例。它核对原API wire提示/原图字节、源码、初始输入、模型配置与runtime，复制历史到新目录并保存来源文件哈希，最多续接一次最终诊断；不新增Active观察或修复。随后首次执行Fixed。原失败目录永久保留incomplete，新结果明确标记continuation，不能称为全程连续/全程流式运行。评分会合并旧失败和新台账，重复来源会拒绝，未知usage/cost不填零；汇总总费用时每个原始API台账只能提供一次。

## 使用自己的参考输入

`single` 是现有 `run_iterative` 的薄入口，Generator、Verifier、修复器仍共用同一个冻结 API 和原生 Chromium harness；不读取本次受控案例或私有评分标签：

```bash
$PY scripts/run_api_pipeline.py single --reference-image reference.png --task-file task.txt \
  --out output/my-api-run --policy active --budget 4 --revisions 1 \
  --max-output-tokens 8192 --request-timeout 300 --max-calls 20 --stream
```

`task.txt` 为公开设计要求。默认初始生成提示自动包含公开任务与 Program ABI；可用 `--generation-prompt` 提供完整公开提示，或 `--initial-program` 从已有候选开始。此入口的 `accepted` 仅为模型判定；独立 GT 评分需单独实现相应任务的公开评分协议。本次 `prepare/run/score` 则固定用于上述三个受控实验案例，不是通用照片 benchmark。
