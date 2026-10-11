# UniPhys E2E：分支进度与后续工作

> 分支：`laq/uniphys-e2e`｜核对基线：`0ebc025`（2026-10-11）
> 依据：[Issue #4 Roadmap](https://github.com/jwj1342/TurnItOver/issues/4#issuecomment-6098267539)及[训练与评测补充](https://github.com/jwj1342/TurnItOver/issues/4#issuecomment-6098859780)。本文记录已证实的实现，不把计划、脚本存在或 Verifier 接受等同于研究成果。

## 给后续 Coding Agent 的约束（必须遵守）

**每次修改代码后、结束本次任务前，必须同步检查并最小化更新本文档。** 只改动受到影响的进展、缺口、优先级和验证证据；不重写无关段落，不累积流水账。即使状态未变化，也要更新文末的「最近同步」一行，注明本次改动及「状态不变」。只有实现和对应验证均可核实时，才将事项写为「已完成」；未运行的测试、未执行的脚本及模型自报 `pass` 不算验收。保持下述任务清单最多五项，完成后删去或替换，不持续追加。

## 目标与现状

本分支要把现有 3D 资产变成**交互式 Verifier 的训练与评测数据**，最终验证：证据约束监督能否让小型视觉模型在有限观察预算下更好地选择视角/关节动作、定位缺陷并及时停止。主线是「资产 → 合格候选与独立真值 → 观察 Episode → 训练监督 → SFT → 留出评测」。**目前工程运行与评测基础已具备，训练—评测研究闭环尚未打通。**

**已实现：资产与运行入口。** [`download_uniphys.py`](../scripts/download_uniphys.py) 下载并校验 UniPhys-Bench Part 2 的指定资产，记录版本和文件哈希；[`render_uniphys_references.py`](../scripts/render_uniphys_references.py) 从 URDF 构造内存 `AssetSpec`，导出多视角参考图，并注明 continuous joint 固定零位、不计作已验证运动覆盖。[`run_uniphys_iterate_batch.py`](../scripts/run_uniphys_iterate_batch.py) 接入 Generator → Render Gate → Verifier → Repair，保留任务状态、调用与 token 统计。已有一次 20 物体真实模型批量运行，**4 个被 Verifier 接受**；这不是独立正确率，详细条件见 [`uniphys-e2e.md`](uniphys-e2e.md)。

**已实现：初版缺陷基准、批跑入口与评分。** [`prepare_uniphys_verifier_benchmark.py`](../scripts/prepare_uniphys_verifier_benchmark.py) 新增 **20 物体 × clean / 关节轴错误 / 部件偏移 = 60 案例**构造器，生成候选代码、私有 gold，并逐案例预检；本地清单确认 60 案例已完整生成且候选程序 SHA 唯一。[`run_uniphys_verifier_batch.py`](../scripts/run_uniphys_verifier_batch.py) 在调用现有 Verifier 前校验 catalog SHA、test-only 划分、60 个唯一 case、每物体三种变体及候选 SHA，逐案例原子记录进度，完成后调用 [`scoring.py`](../turnitover/verifier/scoring.py) 生成指标。结构化输出已含 `joint_id` 与 `evidence_steps`，已有相关单元测试。**但尚未执行这 60 案例的真实模型全量评测并形成正式指标。**

**已实现：资产身份与 test-only 评测边界。** [`convert_uniphys.py`](../scripts/convert_uniphys.py) 将选定 URDF 固化为可由 `CatalogSource` 加载的 `AssetSpec` catalog，逐资产记录数据集 revision、转换输入哈希、许可证、continuous joint 零位固定策略和唯一 `split_group`，并输出全部映射到 test 的显式清单。旧下载清单可逐文件验证其中 14 个资产；其余 6 个资产记录本次实际转换输入哈希，不冒充旧清单已覆盖。60-case 构造器只遍历该 catalog 中的20个 `AssetSpec`，不再重复维护对象 ID 或读取可变的原始 `meta_data.json`，并在 object、case 和 private gold 中记录 `split_group` 与 `split=test`；catalog 版完整产物位于 `data/benchmarks/uniphys-verifier-60-catalog-v1/`。复现命令为 `python scripts/convert_uniphys.py`，随后运行 `python scripts/prepare_uniphys_verifier_benchmark.py`。

**可复用的既有基础。** ReplicaCAD 数据引擎已支持按原始资产分组、腐蚀样本和私有真值隔离；Oracle 动作/成对诊断导出、Qwen3-VL 处理器适配及 assistant-only loss 已存在，见 [`dataset-pipeline.md`](dataset-pipeline.md) 和 [`oracle-supervision.md`](oracle-supervision.md)。这些是可复用组件，**不是 UniPhys 新数据已接通或模型已完成微调**。

## 关键缺口

**评测数据还缺证据资格。** 资产身份、每物体分组和 test-only 边界已经冻结；此前 20 物体批次仅作工程测试，未进入实验，不据此判定数据泄漏。当前构造器仍只有正常及两种单缺陷，尚无组合缺陷、自然生成错误的合格真值、证据充分性标签和按可见历史组织的完整 Episode。自然生成程序部件 ID 与参考不同时，现有几何比对也不能直接充当独立 gold。

**训练与研究对照尚未完成。** 旧动作导出缺停止监督，成对诊断缺 `pass`，且部分相同可见历史对应不同的特权教师动作；`evidence_steps` 引用有效步骤不等于证据足以识别缺陷。尚无实际 SFT 训练入口及 checkpoint，也没有同底座未微调 / 普通 SFT / 证据约束 SFT 比较、fixed / random / active 统一预算实验、监督消融或按物体留出的修复收益结果。

## 下一步（按优先级）

1. **P0｜执行冻结的60-case评测。** 固定 judge 模型、预算、策略和 seed，运行 `python scripts/run_uniphys_verifier_batch.py`；保留逐案例结果、批次清单和 `metrics.json`，中断时以清单中已完成记录为准。**完成标准：**60 个 case 均有终态结果且评分成功，不把脚本存在当作已完成实验。
2. **P0｜做一批有资格的候选与证据标签。** 先复用 clean / axis / offset，补少量多缺陷与自然生成错误；先定义检查范围和需要的证据，再注入或采集错误。独立校验缺陷生效、部件/关节对应关系及 `pass / fail / uncertain`，不要仅凭像素差或 Verifier `pass` 标 gold。**完成标准：**每个候选有可审计的 private gold、证据资格及失败原因。
3. **P0｜统一 Episode 并导出训练监督。** 关联候选版本、可见参考、动作、观察、预算、诊断/停止和独立真值；分别导出「历史 → 下一动作」与「历史 → 诊断或停止」，补齐 `pass`、`uncertain`、停止目标，处理同观测多教师动作。**完成标准：**按资产隔离的数据可直接送入现有 Qwen 处理器，private gold 不进入输入。
4. **P1｜先跑通最小 SFT，再比较监督方案。** 用 Qwen3-VL-2B 优先训练结构化诊断与停止，再加入动作选择；在相同底座、输入、数据规模与预算下比较未微调、普通 SFT、证据约束 SFT。**完成标准：**有可重跑训练入口、配置、曲线、validation 选出的 checkpoint 和训练前后结果。
5. **P1｜完成留出评测与归因。** 先固定观察比较诊断/定位/误放行/停止，再按统一成本比较 fixed、random、active；消融证据约束、停止监督、自然错误数据。闭环固定 Generator，每轮修复后重取证据并独立评分。**完成标准：**结果按原始物体汇总，报告观察成本与波动，能区分诊断、取证、停止及实际修复收益。

优先跑通**小规模且可信的完整链路**，不要在训练前追求覆盖全部 29 种缺陷或扩大到不可审计的数据规模。

---

最近同步：**2026-10-11 · 本次60-case评测准备**；构造器已去除原始元数据与重复对象列表依赖，新增冻结身份/变体去重校验及增量批跑评分入口；相关6项单元测试通过，真实模型全量评测尚未运行，未涉及 ReplicaCAD、训练数据或 SFT。
