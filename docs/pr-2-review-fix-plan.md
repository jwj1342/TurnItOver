# PR #2 Review 最小改动修改计划

## 目标

完成 PR #2 reviewer 提出的代码、测试、文档和证据修正的最小改动方案，使原生照片重建闭环的运行记录可信、错误分类准确、测试覆盖真实执行链路，并让 PR 正文与仓库内证据一致。

## Reviewer 意见对照表

| Reviewer 意见 | 对应计划 | 完成结果 |
| --- | --- | --- |
| 真实模型 E2E 声明与仓库记录、artifact 不一致 | 第 4 项 | 已审计：受控多视图 fixture 上存在一次 `accepted=true` 的真实模型 E2E，证据快照已归档；运行仍对应 dirty 旧提交，声明需限定范围 |
| 文档声称支持不存在的 Viewer | 第 7 项 | 已删除 Viewer 章节和悬空路径，并补充预算作用域与实现边界 |
| PR #1 与 PR #2 被描述为可独立合并，但 README 和 research-status 存在冲突 | 第 9 项 | 已准备准确的 PR 正文措辞；两个 PR 无代码依赖，但后合并者需解决文档冲突 |
| 编译工具缺失等环境问题会错误消耗 runtime repair | 第 2 项 | 主问题已修复（`6f663bb`）；浏览器运行期崩溃分类及 ABI/load、空白渲染测试待补 |
| visual revision 正常耗尽时被记录成 invalid patch | 第 1 项 | 已完成：根终止原因为 `revision_limit`，四类计划内边界测试均已覆盖 |
| invalid patch 后重复 render 相同源码 | 第 5 项 | 源码 SHA 未变化时跳过 gate |
| runtime repair 与 visual revision 的预算作用域未说明 | 第 7 项 | 已明确前者每轮重置、后者全局共享 |
| 新增了三份重复 JSON writer | 第 10.1 项 | 已统一为公共原子 JSON writer |
| 正常 finish reason 取值散落在五处 | 第 10.2 项 | 已统一为公共判断函数 |
| `propose_repair` 与 `run_repair` 存在重复逻辑 | 第 10.3 项 | 已复用公共响应解析 helper，保留 audit 分支 |
| iterate 与 verify 重复声明十个 CLI 参数 | 第 10.4 项 | 已提取共用 argparse 参数注册函数 |
| render gate 用固定文案覆盖浏览器/系统原始错误 | 第 3 项 | 已修复（`c7c351a`）；gate artifact、repair feedback 和根 result 均保留原始错误，定向测试通过 |
| 9 月 12 日 research-status 记录被覆盖 | 第 8 项 | 已恢复历史原文，并将本 PR 当前验证结果另起段落 |
| 架构文档和仓库约定仍称生成闭环暂不做 | 第 7 项 | 已同步 architecture、verifier 和 CLAUDE；README 原表述无需修改 |
| 主流程测试完全 mock render gate | 第 6 项 | 已新增真实经过 gate 的 browser 集成测试 |
| `.gitignore` 新增了无关的 `.venv-openhands/` | 第 11 项 | 已删除该忽略项；本地虚拟环境保持未跟踪且不纳入 PR |
| 数据引擎测试被称为主分支既有失败，但 reviewer 尚未确认 | 第 12 项 | 已复核当前环境：shard 可复现性用例连续 3 次通过；撤回历史失败归因 |
| Reviewer 将提交 `repair/loop.py` 的新改动 | 实施前准备、第 10.3 项 | 先同步最新主分支，再完成 repair 去重 |

## 实施顺序

1. 同步最新 `origin/main`，吸收 reviewer 对 `repair/loop.py` 的新改动。
2. 完成 P0：终止原因、环境错误、原始错误和 E2E 证据。
3. 完成 P1：跳过重复 render、真实 gate 集成测试和文档修正。
4. 完成 P2：公共逻辑去重、无关改动清理和全量测试。
5. 根据最终代码、artifact 和测试结果更新 PR 正文。

## 实施前准备

- Fetch 最新 `origin/main`，确认 PR head 和 merge base。
- 检查 reviewer 提到的 `repair/loop.py` 改动是否已经进入主分支；先整理分支，再做 repair 公共逻辑重构。
- 检查 PR #1 与 PR #2 在 README 和 `docs/research-status.md` 上的冲突，确定实际合并顺序。
- 记录当前单元测试、浏览器测试和数据引擎 shard 可复现性用例的基线结果。

完成条件：PR #2 基于最新主分支，`loop.py` 的上游改动已保留，PR #1 的文档冲突位置和测试基线已经明确。

## P0

### 1. 修复 `max_visual_revisions` 终止原因

涉及文件：

- `turnitover/repair/iterative.py`
- `tests/unit/test_iterative_synthesis.py`

修改方案：

- visual revision 总预算耗尽时，根 `result.json` 使用：
  - `status=max_visual_revisions`
  - `termination=revision_limit`
- `invalid_patch` 只记录在单次 `visual_revisions[*].outcome` 中，不再作为正常预算耗尽的根终止原因。
- generator 主动停止继续记录为 `termination=generator_stop`。

测试：

- `max_visual_revisions=0` 时 verifier fail。
- 一次有效 revision 后 fresh verifier 再次 fail。
- 连续 invalid patch 后预算耗尽。
- generator 主动 stop。

完成条件：所有预算耗尽路径都记录为 `revision_limit`，单次 patch 失败信息仍完整保留。

当前进度（2026-10-07）：`462c72b` 已修正根终止原因；现已覆盖 `max_visual_revisions=0`、一次有效 revision 后再次 fail、连续 visual `invalid_patch` 和 `generator_stop` 四类路径。

### 2. 完善 environment/system error 分类

涉及文件：

- `turnitover/render/gate.py`
- `turnitover/render/transpile.py`
- `turnitover/repair/iterative.py`
- 对应单元测试和浏览器测试

修改方案：

- 为 render gate 结果增加明确的失败归因，例如 `failure_kind=candidate|environment`。
- `stage` 只记录发生阶段，不再兼任错误归因。
- 以下问题归为 environment，直接终止：
  - esbuild 不存在、无执行权限或无法启动；
  - 浏览器或 Playwright 启动失败；
  - 浏览器崩溃及系统级 I/O 异常。
- 以下问题归为 candidate，可进入 runtime repair：
  - TypeScript 源码编译失败；
  - Program ABI 或候选加载失败；
  - 空白渲染。
- environment 错误结束为 `termination=environment_error`，不增加 `model_calls.runtime_repair`。

测试：

- 缺失 esbuild 不调用 repair。
- 浏览器启动异常不调用 repair。
- 非法 TypeScript 会进入 repair。
- ABI/load 错误和空白渲染会进入 repair。

完成条件：环境问题不消耗生成模型修复次数，候选程序问题仍能正常修复。

当前进度（2026-10-07）：`6f663bb` 已增加 `failure_kind`，验证缺失 esbuild 直接以 `environment_error` 终止且不调用 repair。未携带 harness 结构化错误的 Playwright load/request-view 失败现在归为 `BrowserTransportError` 与 environment；结构化 ABI/load 错误和空白渲染均归为 candidate 并进入 runtime repair。对应单元与真实 browser 回归测试已通过。

### 3. 保留 render gate 原始错误

涉及文件：

- `turnitover/render/gate.py`
- `turnitover/repair/iterative.py`
- render gate 与 iterative 测试

修改方案：

- 删除固定错误文本 `Browser environment could not complete the render gate.`。
- render gate artifact 保存原始：
  - `error_type`
  - `message`
  - `stage`
  - `failure_kind`
- candidate 错误通过 `_render_feedback` 原样传给 repair 模型。
- environment 错误直接终止，但在根 artifact 中保留同样的错误详情。

测试：

- 构造带唯一消息的异常，检查 gate artifact 和根 artifact。
- candidate 错误的 repair feedback 必须包含相同类型和消息。
- environment 错误必须保留信息且不调用 repair。

完成条件：运行记录中可以看到原始错误，repair 模型不会再收到模糊固定文案。

当前进度（2026-10-07）：`c7c351a` 已保留原始错误，并由定向测试确认 gate artifact、candidate repair feedback 和 environment 根 `result.json` 中的信息一致。第 1–3 项相关定向测试共 11 项通过，仍需最终全量测试。

### 4. 统一真实模型 E2E 声明与证据

涉及内容：

- 本地 `output/iterate-*` 候选运行
- 版本化 results artifact
- PR 正文
- README、`docs/research-status.md`、`docs/iterative-synthesis.md`

修改方案：

先审计已有运行 artifact，检查：

- 是否能对应到明确代码版本；
- 是否包含 generation、render gate、verifier trajectory/model calls 和根 `result.json`；
- 是否能证明生成了合法 Program ABI，并完成正文所称的四步主动观测；
- 输入、日志和响应是否适合提交；
- 最终终态是 accepted、fail、uncertain 还是预算耗尽。

根据审计结果选择：

1. 证据完整：提交最小完整 results 快照和 manifest，保留 E2E 声明，并准确写明实际终态，不把“链路跑完”写成“最终通过”。
2. 证据不完整：删除 PR 正文中的 E2E 已跑通声明，统一改为“单元测试和真实浏览器测试验证了工程链路，未提交可审计的真实模型 E2E 结果”。

当前进度（2026-10-07）：`output/smoke-test/pr2-pipeline-no-thinking` 包含最强证据链：真实 generation、render gate、4 次 judge 调用和三步主动观测，根结果为 `accepted=true`，verifier 返回 `pass`。输入是受控 toy cabinet 多视图拼图，运行记录对应 `410ce57` 且 `dirty=true`；证据快照已归档到仓库。因此可以声明“受控 fixture 上的真实模型 E2E 已落地”，但不能扩展为当前 clean commit、单张真实照片或泛化效果结果。

PR 正文改为：“单元测试和真实浏览器测试验证了工程链路；另有一次受控多视图 fixture 的真实模型 E2E 运行完成 generation、render gate 和 active verification，并以 `accepted=true` 结束。该 artifact 已版本化归档，但运行本身对应 dirty 的旧提交；不作为真实照片或泛化效果结果。”

完成条件：PR 正文、仓库文档和版本化 artifact 三者一致，每项真实模型声明都可以直接复核。

## P1

### 5. 无效 patch 时跳过重复 render

涉及文件：

- `turnitover/repair/iterative.py`
- `tests/unit/test_iterative_synthesis.py`

修改方案：

- runtime repair 前后比较 `program.sha`。
- invalid patch 或相同 SHA 的 no-op patch 只记录 repair attempt，不重新执行 gate。
- 只有新源码 SHA 发生变化时才创建下一个 render-gate attempt。
- repair 模型调用、token 和失败明细仍正常计数。

测试：

- 两次 invalid patch：repair 2 次，gate 仅 1 次。
- 第一次 invalid、第二次有效：repair 2 次，gate 2 次。
- no-op patch 不触发新 gate。

完成条件：相同源码不会被重复渲染，调用和 artifact 计数准确。

### 6. 增加真实经过 render gate 的主流程集成测试

涉及文件：

- `tests/browser/test_iterative_synthesis.py`，或现有 browser 测试中的同类文件

修改方案：

- 直接调用 `run_iterative`。
- 使用真实 `run_render_gate`、esbuild、Playwright 和浏览器 harness。
- 使用 toy Program ABI 候选与本地生成的 reference fixture。
- verifier/model decision 可以使用确定性 fixture，但不能 mock render gate。
- 检查 `render.png`、gate `result.json`、program SHA、artifact 路径和根终态。

完成条件：至少一个 browser 测试真实经过 `run_iterative -> render gate -> verifier -> result.json`，且不访问外部模型。

当前进度（2026-10-07）：已新增 `tests/browser/test_iterative_synthesis.py`。测试使用真实 esbuild、Playwright、render gate 和 verifier，仅以本地确定性 generator/judge fixture 代替外部模型；与既有 render gate 测试一起在真实 Chromium 环境通过。

### 7. 修正文档与实现不一致

涉及文件：

- `docs/iterative-synthesis.md`
- `docs/architecture.md`
- `docs/verifier.md`
- `CLAUDE.md`
- README 和 `docs/research-status.md`

修改方案：

- 删除不存在的 Viewer 支持章节和路径。
- 明确两类预算：
  - `max_runtime_repairs` 在每个 visual round 重新计算；
  - `max_visual_revisions` 在整次运行中全局共享。
- 将“生成—修复闭环暂不实现”更新为：原生推理闭环已实现；训练、正式真实照片评测和效果结论尚未完成。
- 保留只针对历史实验范围的限制，不把所有“未包含生成修复”描述机械删除。

完成条件：用户文档、架构文档和实际实现一致，不再引用未提交的 Viewer，也不扩大实验结论。

当前进度（2026-10-07）：已删除 Viewer 章节，明确 runtime repair 每个 visual round 重置、visual revision 全局共享，并同步修正 architecture、verifier 和 CLAUDE 中的过期范围描述。

### 8. 恢复 `research-status` 历史记录

涉及文件：

- `docs/research-status.md`

修改方案：

- 从最新主分支恢复 9 月 12 日“本次交接检查”的原文和原有测试数字。
- 在其后新增“PR #2 原生迭代闭环验证”段落。
- 新段落记录本轮实际单元测试、浏览器测试、真实 gate 集成测试、E2E artifact 情况和 `git diff --check` 结果。

完成条件：9 月 12 日记录未被改写，本 PR 验证结果独立追加且可由测试输出复核。

当前进度（2026-10-07）：已恢复主分支的 9 月 12 日原文，并另起“PR #2 原生迭代闭环验证”段落；完整非浏览器测试已通过（1 项跳过），完整浏览器测试 16 项通过。

### 9. 处理 PR #1/PR #2 文档冲突

涉及内容：

- PR 正文
- README
- `docs/research-status.md`

修改方案：

- 检查 PR #1 先合并和 PR #2 先合并两种顺序的实际冲突。
- PR #2 继续保持代码上不依赖 PR #1 的 Viewer/handoff 文件。
- 将“两个 PR 可独立合并”改为：两个 PR 没有代码依赖，但修改了相同文档，后合并的一方需要解决文档冲突。
- 按实际合并顺序同时保留原型交接说明、正式闭环说明和各自验证记录。

完成条件：正文准确描述两个 PR 的关系，计划采用的合并顺序不丢失 README 或 research-status 内容。

当前进度（2026-10-07）：已确认两个分支同时修改 README 和 `docs/research-status.md`。PR 正文改为：“PR #1 与 PR #2 没有代码依赖，可分别 review；两者修改了相同文档，后合并的一方需要解决文档冲突并保留双方说明与验证记录。”当前分支不引入 PR #1 的 handoff 文件或悬空链接。

## P2

### 10. 整理重复代码

#### 10.1 JSON writer

- 提取一个公共原子 JSON writer。
- 统一 UTF-8、缩进、`ensure_ascii=False`、`allow_nan=False` 和临时文件替换。
- 替换 `generation.py`、`render/gate.py`、`repair/iterative.py`、`verifier/runner.py` 的重复实现。
- 不改变 JSON schema 和 artifact 目录。

#### 10.2 模型 finish reason

- 在 `turnitover/models` 中定义唯一正常结束集合或判断函数。
- `generation.py`、`models/commands.py`、`verifier/policies.py`、`repair/loop.py` 统一复用。
- 原始 provider finish reason 继续写入 artifact。

#### 10.3 `run_repair` / `propose_repair`

- 基于 reviewer 更新后的最新 `repair/loop.py` 实施。
- 提取公共响应解析 helper，复用 finish reason 校验、响应 metadata 落盘与 patch 解析；保持两条流程各自的请求准备和 artifact 时机。
- 保留 `run_repair` 独有的 observe、私有 audit、评分和候选接受逻辑。
- 保留 `expected_first_request` 在模型调用前进行 prompt/图片一致性校验。

#### 10.4 CLI 参数

- 提取 verify/iterate 共用的 argparse 参数注册函数。
- 仅合并真正相同的 task、policy、budget、seed、size、views、actions、runtime threshold 和 env 参数。
- 保持 `--reference-image` 的 required 差异及两个命令的专属参数。
- 补 parser 测试，检查默认值、choices 和 required 语义。

完成条件：四类重复代码均有唯一实现，现有行为和 artifact 契约不变。

当前进度（2026-10-07）：`eb64cf3` 已新增 `core/jsonio.py` 并替换四处原子 JSON writer；五处 finish reason 判断统一为 `models/client.py` 的公共判断；repair 共享响应解析 helper，`run_repair` 的首请求校验、私有 audit、评分与 ledger 保持不变；verify/iterate 复用参数注册函数且保留各自专属参数与 reference image required 差异。新增 JSON、finish reason 与 CLI 回归测试，完整非浏览器测试通过（1 项跳过），完整浏览器测试 16 项通过。

### 11. 清理无关改动

涉及文件：

- `.gitignore`

修改方案：

- 删除本 PR 新增的 `.venv-openhands/` 忽略项。
- 检查最终 diff，排除 handoff Viewer、虚拟环境、缓存、日志和其他无关文件；仅保留已归档的 `output/smoke-test/pr2-pipeline-no-thinking/` 作为可审计 E2E 证据。

完成条件：PR 不再包含 `.venv-openhands/` 改动及其他无关文件。

当前进度（2026-10-07）：已删除 `.venv-openhands/` 忽略项；本地虚拟环境目录不暂存、不提交。`output/smoke-test/pr2-pipeline-no-thinking/` 是有意归档的 E2E 证据，最终 diff 仍需在全部修改完成后复核。

### 12. 复核数据引擎 shard 可复现性用例

修改方案：

- 确认 `test_catalog_clean_and_mixed_are_shard_independent` 检验“整体生成”和“两 shard 生成后合并”逐字节一致的可复现性契约。
- 记录它是依赖 Chromium 与已构建 `web/dist` 的 browser 用例，覆盖 `turnitover/engine/generate.py`，不覆盖本 PR 新增的 `iterate` 路径。
- 若当前环境连续通过，删除“当前环境失败”及“`origin/main` 同样失败”的历史归因；不把尚未复现的环境解释写成数据引擎根因。

完成条件：PR 正文只陈述可复核的当前结果，不再包含未经确认的失败归因。

当前进度（2026-10-07）：在 `web/dist` 与 Chromium 已就绪的当前环境中，`test_catalog_clean_and_mixed_are_shard_independent` 连续运行 3 次通过。此前本机失败的记录已经过时；由于未对历史失败完成可复现的根因定位，不将其归因为主分支或数据引擎逻辑问题。

### 13. 完整验证

按以下顺序执行：

```bash
# 定向测试
python -m pytest tests/unit/test_iterative_synthesis.py tests/unit/test_repair.py

# 完整非浏览器测试
.venv-local/bin/python -m pytest -m 'not browser' -q

# 完整浏览器测试
.venv-local/bin/python -m pytest -m browser -q

# Diff 检查
git diff --check origin/main...HEAD
git diff --check
```

同时检查：

- `invalid_patch` 不再用于正常预算耗尽的根 termination；
- finish reason 只保留一个公共定义；
- 重复 JSON writer 已移除；
- Viewer 和“生成闭环暂不做”的过期描述已清理；
- Markdown 相对链接有效；
- 最终 diff 只包含 review 要求的修改。

完成条件：完整单元测试、浏览器测试和 `git diff --check` 通过，`research-status` 中的测试数字与实际输出一致。

当前进度（2026-10-07）：完整非浏览器测试已通过（1 项跳过）；完整浏览器测试以项目 Playwright 缓存运行，16 项通过。归档 artifact 保留模型原始文本中的尾随空白，最终 diff whitespace 检查须将该证据目录作为保真例外单独记录，其余代码和文档 diff 必须通过检查。

## 最终交付

- Reviewer 意见逐项完成对照表。
- 各项代码、测试和文档修改摘要。
- E2E artifact 审计结论及最终声明。
- PR #1/PR #2 冲突处理结果。
- 数据引擎 shard 可复现性用例的当前复核结果。
- 完整测试与 `git diff --check` 输出。
- 与最终结果一致的 PR 正文。
