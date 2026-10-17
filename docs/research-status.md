# 合作者入口：当前在验证什么（2026-09-12）

我们研究的问题是：**验证器能否通过主动转动关节、改变视角来取得缺陷证据，并用更好的反馈帮助冻结生成器修复三维程序？** RP 的路线是训练小型视觉语言验证器，生成器保持冻结。当前阶段先做数据资格检查、能力拆解和修复试验，判断训练应该解决什么问题；尚未训练模型。

建议先读本文，再看[能力诊断](capability-diagnosis.md)和[修复实验](repair-experiment.md)。
[RP](../RP.md) 是研究设想；本文记录已验证进展与尚未成立的假设，不把规划当作结果。

正式 `turnitover` 包新增原生照片重建迭代入口 `turnitover iterate`，复用 Program ABI、`ObservationSession`、verifier 与 patch protocol：单次生成候选 → 确定性 render gate → 证据驱动的有限轮 verifier 修复，render 失败与视觉修改各有独立预算，只有 evidence-linked 的 verifier pass 才算接受。该路径经单元与浏览器 fixture 验证，但尚未运行真实照片模型实验，不构成新的效果结果。参见[原生照片重建迭代闭环](iterative-synthesis.md)。

## 已打通的流程

1. 获取有来源记录的 ReplicaCAD 关节资产，转换为可执行 Three.js 程序，并用原始 URDF 独立核对运动。
2. 注入已知偏移/轴错误，生成参考与候选，按资产族划分数据，隔离模型输入和私有真值。
3. 渲染有限观测状态，构造缺陷暴露代理、预算内 oracle/固定/随机轨迹，导出动作和成对诊断监督。
4. 用冻结 Qwen 分别测动作选择、差异辨别、缺陷分类与完整协议，避免将一个失败笼统归因于视觉能力。
5. 用 OpenRouter GPT-5.6 Sol 在相同案例上比较四种反馈，执行真实补丁并独立验收。

模型输入只含该条件允许的参考图、候选观察及反馈。参考源码与完整审计属于评测真值；
只有显式 gold 条件会提供失败约束。检查器读取浏览器导出的证据，不直接读取腐蚀参数来判通过。

## 证据与可支持的结论

| 问题 | 已有证据 | 当前判断 |
| --- | --- | --- |
| 资产转换是否保留运动？ | 6 资产、102 状态的独立 FK/拓扑审计通过，最大顶点误差约 3.10e-7 m | 可以用于当前结构/运动学实验；不覆盖物理与完整材质 |
| 观测选择有无潜在价值？ | 全 96 样本、120 缺陷，六步暴露 oracle/固定/随机为 115/76/66 | 特权观测选择有价值；暴露是像素代理，不等于模型识别 |
| 测试集能否证明主动优势？ | 16 个测试样本都来自一个门资产族，oracle/固定同为 23/23 暴露 | 不能；需要多关节、多个留出资产族 |
| 小模型具体卡在哪里？ | Qwen 主动策略 47 步全部请求 front；最明显图对辨别正确 14/16，但完整诊断无正确缺陷检出 | 动作、类别理解和协议均需单独验证；不能断言模型没有视觉能力 |
| 生成器能否执行有效修复？ | Sol 固定观察修复 1/3，约束/oracle/联合各 3/3；干净对照均保持 | 在这些案例上能够修复，反馈值得研究；不能外推训练收益 |
| 训练数据是否就绪？ | 576 条动作、288 条成对诊断；处理器与答案掩码已验证 | 格式链路就绪，监督完整性与可学习性仍不充分 |

这些数字的分母不同：96 是全数据工程诊断；16 是同族测试样本；修复试验每条件只有 3 个损坏案例和 1 个干净对照。不能合并成统一准确率。Qwen 后续提示诊断是在看过测试输出后追加的，属于开发诊断，不是最终盲测。

修复批量实际 30 次调用，供应商返回费用 $3.444346；独立 smoke 另计 $0.1313835。
固定观察在两例未通过时提前停止，所以调用更少不代表修复更高效。
模型标识和请求参数固定，但未固定供应商权重快照或随机种子。

## 为什么现在不直接扩大微调

同一可见前缀对应不同的特权教师动作，已发现 5 组、涉及 36 条记录。导出的经验动作分布
能表达这种冲突，却不证明教师动作在单参考条件下可推断。成对诊断又额外提供了匹配状态的
正确参考，不能直接当作单参考部署任务的监督。此外，还缺少单参考的最终诊断、停止与通过监督。

当前修复 oracle 固定初始腐蚀的观察序列，未在每次修改后重算；gold 反馈是确定性几何/预算
约束，不是 RP 的完美语义诊断。因此尚未完成 RP 的三条 oracle 上界，也不能套用其中
“效率提升不足两成则调整路线”的决策阈值。

## 下一轮按什么顺序做

| 顺序 | 工作 | 完成标准 |
| --- | --- | --- |
| 1 | 扩展多关节与多个独立留出资产族，冻结拆分与评估提示 | 固定预算不能轻易遍历所有关节；结果按资产族报告，开发集与最终集分离 |
| 2 | 对每轮候选重建观测/诊断 oracle | 修改后的证据重新验证；新增、消除或转移的错误被记录；不把初始腐蚀标签永久当答案 |
| 3 | 重复采样四条件修复试验 | 预先固定调用预算、失败处理和成功/停止指标；报告所有重复与资产间差异 |
| 4 | 补齐可见条件下的动作、类别、通过/停止监督，再做小规模训练 | 干净、难负例和证据不足样本齐全；同输入冲突有明确处理；在新留出集比较训练前后 |

这是后续实验计划，不是本次结果。尚无训练 checkpoint，也未证明物理正确性或真实照片泛化。

## 克隆后怎样检查与复现

无需资产、GPU 或 API 即可阅读[已提交结果快照](../results/2026-09-12/README.md)，包括原始指标、
修复响应、逐轮审计和费用。快照约 270 KB，不含完整程序、图片和训练数据，不能单凭它重放渲染。

安装环境后，可以离线重算修复汇总：

```bash
source scripts/setup_local.sh
python scripts/summarize_repair_experiment.py results/2026-09-12/output/repair-sol-pilot-v1
pytest
pytest -m browser
```

完整重建顺序如下。所有命令从仓库根目录运行，输出目录应不存在；第二次执行应更换输出路径及
下游对应参数。下载和 API 步骤需联网；集群上超过 16 样本的生成/渲染任务应通过 Slurm。

```bash
source scripts/setup_local.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/download_replicacad.py
python scripts/convert_replicacad.py
python scripts/audit_replicacad.py --out output/replicacad-fk-v3
# 以下完整数据任务在集群上提交到计算节点；个人机器可直接运行。
python scripts/run_mesh_benchmark.py
python scripts/run_oracle_supervision.py --out data/benchmarks/oracle-v1
python scripts/export_paired_diagnosis.py --oracle data/benchmarks/oracle-v1 --out data/benchmarks/paired-v1
python scripts/export_action_distributions.py --source data/benchmarks/oracle-v1 --out data/benchmarks/oracle-action-distributions-v1
python scripts/prepare_repair_experiment.py --out output/repair-prepared-new
```

实际模型修复会产生 API 费用。按 [.env.example](../.env.example) 在本地配置 generator：
provider 为 `openrouter`，model 为 `openai/gpt-5.6-sol`，max tokens 为 `8192`，timeout 为 `180`。
密钥由各自环境提供。然后执行：

```bash
python scripts/run_repair_experiment.py --prepared output/repair-prepared-new --out output/repair-model-new
python scripts/summarize_repair_experiment.py output/repair-model-new
```

输入构造可重建，API 响应不保证逐字复现。Qwen 环境和冻结模型配置见[本地模型记录](verifier-smoke.md)，
评估入口为 `scripts/evaluate_*qwen.py`、`scripts/evaluate_pair_readout.py` 和
`scripts/evaluate_defect_readout.py`；这些需本地权重与 GPU，不是上述离线汇总的依赖。

## 本次交接检查

本次提交前执行：102 项单元测试通过，1 项可选 torch 依赖测试跳过；12 项实际浏览器测试通过。
新增回归检查核对全部快照哈希、从逐轨迹记录重算修复汇总，并拒绝汇总与轨迹不一致的结果。
本地文档链接与 `git diff --check` 通过。此次整理没有新增付费模型调用，也没有重新选择或剔除实验案例。

## PR #2 原生迭代闭环验证（2026-10-07）

本轮定向执行 `tests/unit/test_iterative_synthesis.py` 和 `tests/browser/test_render_gate.py`，共 11 项通过；
覆盖 Program ABI 生成、render gate、环境错误、fresh verification，以及视觉修改预算的四类终止路径。
新增 browser 集成测试已真实经过 `run_iterative -> render gate -> verifier -> result.json`；完整非浏览器测试已通过（1 项跳过），完整浏览器测试 16 项通过。
补充的单元回归通过模拟 session 覆盖 Playwright load/request-view 传输失败、Program ABI/load 失败和空白渲染：前者直接作为 environment 终止，后两者作为 candidate 进入 runtime repair；真实 Chromium 用例另行覆盖正常 render gate 与迭代闭环，不将模拟传输异常表述为真实浏览器复现。
`eb64cf3` 统一了原子 JSON writer、模型正常结束判断、repair 响应解析和 verify/iterate 共同 CLI 参数；原有 artifact 路径、repair audit/评分边界和命令专属参数保持不变。
数据引擎 shard 可复现性用例 `test_catalog_clean_and_mixed_are_shard_independent`（覆盖 `turnitover/engine/generate.py`）在已构建 `web/dist` 与 Chromium 就绪的环境连续运行 3 次通过；它不覆盖本 PR 的 `iterate` 路径。此前本机失败未完成可复现根因定位，故不作主分支或数据引擎逻辑失败归因。

本地 `output/smoke-test/pr2-pipeline-no-thinking` 已记录一次 `accepted=true` 的真实模型 E2E：
generation、render gate 和 active verifier 均完成，verifier 在三步主动观测后返回 `pass`。
该记录使用受控的 toy cabinet 多视图输入，对应 `410ce57` 且 `dirty=true`，证据快照已纳入 Git；
该次运行的 runtime repair 与 visual revision 调用均为 0，因此只证明受控 fixture 上的单轮生成—渲染—验证成功，不构成多轮修复效果、当前 clean commit、单张真实照片重建或泛化效果结果。

## PR #2 复核修正（2026-10-10）

本轮按 reviewer 的逐行复现修正三处边界：空白图指标只统计空间变化；generation、runtime repair 和 visual revision 在实际模型调用边界记录失败调用及已知 token；fixed/random 的非对象 JSON 统一进入非法决策反馈与最多三次尝试。新增的真实 Chromium 用例使用确定性模型 fixture，完整经过 verifier fail、`propose_repair`、补丁应用、第二轮 render gate 和 fresh verifier pass。

在最新代码提交上重新构建 `web/dist` 并执行 typecheck；完整非浏览器测试为 141 项通过、1 项跳过，完整浏览器测试为 18 项通过。环境为 Python 3.12.7、Node.js 22.22.3、npm 10.9.8、Playwright 1.57.0、Chrome for Testing 143.0.7499.4。reviewer 提到的 `repair/loop.py` 本地改动尚未推送，当前保持待同步，不预先声称已吸收。
