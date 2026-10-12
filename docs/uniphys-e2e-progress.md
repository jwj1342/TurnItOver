# UniPhys E2E：分支进度与后续工作

> 分支：`laq/uniphys-e2e`｜核对基线：`0ebc025`（2026-10-11）
> 依据：[Issue #4 Roadmap](https://github.com/jwj1342/TurnItOver/issues/4#issuecomment-6098267539)及[训练与评测补充](https://github.com/jwj1342/TurnItOver/issues/4#issuecomment-6098859780)。本文记录已证实的实现，不把计划、脚本存在或 Verifier 接受等同于研究成果。

## 给后续 Coding Agent 的约束（必须遵守）

**每次修改代码后、结束本次任务前，必须同步检查并最小化更新本文档。** 只改动受到影响的进展、缺口、优先级和验证证据；不重写无关段落，不累积流水账。即使状态未变化，也要更新文末的「最近同步」一行，注明本次改动及「状态不变」。只有实现和对应验证均可核实时，才将事项写为「已完成」；未运行的测试、未执行的脚本及模型自报 `pass` 不算验收。保持下述任务清单最多五项，完成后删去或替换，不持续追加。

## 目标与现状

本分支要把现有 3D 资产变成**交互式 Verifier 的训练与评测数据**，最终验证：证据约束监督能否让小型视觉模型在有限观察预算下更好地选择视角/关节动作、定位缺陷并及时停止。主线是「资产 → 合格候选与独立真值 → 观察 Episode → 训练监督 → SFT → 留出评测」。**目前工程运行与评测基础已具备，训练—评测研究闭环尚未打通。**

**已实现：资产与运行入口。** [`download_uniphys.py`](../scripts/download_uniphys.py) 下载并校验 UniPhys-Bench Part 2 的指定资产，记录版本和文件哈希；[`render_uniphys_references.py`](../scripts/render_uniphys_references.py) 从 URDF 构造内存 `AssetSpec`，导出多视角参考图，并注明 continuous joint 固定零位、不计作已验证运动覆盖。[`run_uniphys_iterate_batch.py`](../scripts/run_uniphys_iterate_batch.py) 接入 Generator → Render Gate → Verifier → Repair，保留任务状态、调用与 token 统计。已有一次 20 物体真实模型批量运行，**4 个被 Verifier 接受**；这不是独立正确率，详细条件见 [`uniphys-e2e.md`](uniphys-e2e.md)。

**已实现：正式 Test 缺陷基准、批跑入口与评分。** [`prepare_uniphys_verifier_benchmark.py`](../scripts/prepare_uniphys_verifier_benchmark.py) 生成 **20 物体 × clean / 关节轴错误 / 部件偏移 = 60 案例**、私有 gold 和逐案例预检；正式产物位于 `data/benchmarks/uniphys-verifier-test/`，60 个候选程序 SHA 唯一。[`run_uniphys_verifier_batch.py`](../scripts/run_uniphys_verifier_batch.py) 在调用现有 Verifier 前校验 catalog SHA、test-only 划分、每物体三种变体及候选 SHA，逐案例原子记录进度，完成后调用 [`scoring.py`](../turnitover/verifier/scoring.py) 生成指标。旧20物体/60案例已归入 `data/benchmarks/uniphys-verifier-dev/`，只作开发回归。**正式 Test 尚未执行真实模型评测。**

**已实现：正式 Test 身份与未见边界。** [`select_uniphys_test.py`](../scripts/select_uniphys_test.py) 排除 `UPB_00000000–UPB_00000019`、旧 Test 全部对象和现有 Train/Validation，再按固定 seed、来源身份、完整模型 SHA 与规范化对象名称筛选20个对象；选择过程不渲染或人工查看候选。冻结清单位于 [`uniphys-test.json`](../configs/uniphys-test.json)，[`convert_uniphys.py`](../scripts/convert_uniphys.py) 默认从该清单生成 `data/assets/uniphys-test-catalog/`，记录 revision、输入哈希、许可证、continuous joint 策略和唯一 `split_group`。程序化审计确认正式 Test 与开发、Train、Validation 的对象交集均为0，三类近重复信号交集也均为0。

**已实现：UniPhys Part 2 Episode 工程验证划分。** [`select_uniphys_trainval.py`](../scripts/select_uniphys_trainval.py) 按固定 seed 执行 URDF 条件检查和完整 Mesh 转换预检，收集内部仍标作100个 Train 与20个 Validation 的对象；这两个名称只用于验证 Episode 生成、隔离与加载链路，冻结配置标记为 `episode_engineering_validation`、`formal_training=false`，**不作为正式 SFT 训练数据**。正式 Test 随后以这120个对象为排除集合，保证对象无交叉。现有工程产物含300个 Train case、60个 Validation case和360份逐案例预检；[`run_uniphys_verifier_batch.py`](../scripts/run_uniphys_verifier_batch.py) 可复用冻结身份检查，但结果只用于工程验证。

**已实现：无模型的统一 Episode 导出代码。** [`generate_uniphys_supervision.yaml`](../configs/generate_uniphys_supervision.yaml) 复用既有生成器与 checker，配置 clean、单缺陷及最多两个组合缺陷；[`episodes.py`](../turnitover/oracle/episodes.py) 将既有 Oracle 轨迹导出为同一 `episode_id` 下的下一动作监督和唯一终止 verdict，按反事实像素暴露产生 `complete / partial / none` 证据资格，并覆盖受限 `pass`、`fail` 与 `uncertain`。[`run_oracle_supervision.py`](../scripts/run_oracle_supervision.py) 仅在既有动作/观察生成之后追加 `*.episodes.jsonl`，不改变生成器、quality、Oracle 规划、Verifier 或 tokenizer。相关单元测试已通过；480样本 UniPhys 数据尚未实际生成。

**可复用的既有基础。** ReplicaCAD 数据引擎已支持按原始资产分组、腐蚀样本和私有真值隔离；Oracle 动作/成对诊断导出、Qwen3-VL 处理器适配及 assistant-only loss 已存在，见 [`dataset-pipeline.md`](dataset-pipeline.md) 和 [`oracle-supervision.md`](oracle-supervision.md)。这些是可复用组件，**不是 UniPhys 新数据已接通或模型已完成微调**。

## 关键缺口

**Episode 数据尚未实际生成。** 正式 Test 已重新冻结且未运行模型；旧20物体批次及其60案例只作开发回归，不进入正式指标。代码现已支持可控组合缺陷、证据资格和统一 Episode，但尚未在 UniPhys Part 2 工程验证划分上执行并核验实际分层计数。该120物体划分即使生成成功也不升级为正式训练集。自然生成错误不属于本次无模型闭环；其部件 ID 与参考不同时，现有几何比对仍不能直接充当独立 gold。

**UniPhys-40K 第一阶段已建立首批合格训练资产，但覆盖仍很小。** 官方 `UniPhysGen@3f1bb2d` 代码确认发布 OBJ 为 `qpos=0`、轴/枢轴使用 OBJ 坐标、旋转范围为弧度、滑动范围为 OBJ 单位；[`convert_uniphys40k.py`](../scripts/convert_uniphys40k.py) 据此逐资产推导尺度，按三角形坐标多重集验证完整模型闭合，并核验来源许可及正式 Test 的来源/OBJ SHA/名称重叠。对40个简单可动候选的严格预检得到5个 structural candidate、35个拒绝；三视图真实浏览器运动检查后，`UP_00015016`、`UP_00015020`、`UP_00039007`、`UP_00039025` 共4个晋级首批 formal Train/Validation catalog（3 Train、1 Validation），`UP_00023000` 因发布打开姿态再旋转135°穿到机身下方而保留为运动失败。所有失败原因及 smoke SHA 均保留；HSSD 网格为 CC BY-NC 4.0，PartNet/ShapeNet 网格受非商业条款约束，均未被 UniPhys 注释的 CC BY 4.0 覆盖。

## 下一步（按优先级）

1. **P0｜执行正式 Test 的60-case评测。** 固定 judge 模型、预算、策略和 seed，运行 `python scripts/run_uniphys_verifier_batch.py`；保留逐案例结果、批次清单和 `metrics.json`，中断时以清单中已完成记录为准。**完成标准：**60 个 case 均有终态结果且评分成功，不混入开发回归结果。
2. **P0｜生成并验收 Part 2 工程验证 Episode。** 先以少量对象 smoke，再通过 Slurm 执行冻结的480样本配置；核对 clean / offset / axis / combined 分层、隔离原因、证据资格、每条轨迹唯一终止目标及内部 train/validation 资产隔离。**完成标准：**产物清单 complete，所有公开 messages 不含 private gold，并可由现有 Qwen 处理器读取；该结果不作为正式 SFT 数据。
3. **P0｜处理同观测多教师动作。** 在新 Episode 上按实际图像 SHA 和可见上下文重新审计动作歧义，复用既有经验分布导出或建立仅使用可见信息的教师。**完成标准：**训练时不会把相同输入的冲突动作当作彼此排斥的唯一标签。
4. **P1｜扩充 UniPhys-40K 正式 Train/Validation。** 复用现有40资产预检与多视图审核门槛扩大来源/机制覆盖，保持 structural candidate 与 training eligible 分离；不得用轴反转、截断范围或未审核资产补数量。首批4资产只证明链路可行，不足以支持大规模训练。
5. **P1｜完成留出评测与归因。** 先固定观察比较诊断/定位/误放行/停止，再按统一成本比较 fixed、random、active；消融证据约束、停止监督、自然错误数据。闭环固定 Generator，每轮修复后重取证据并独立评分。**完成标准：**结果按原始物体汇总，报告观察成本与波动，能区分诊断、取证、停止及实际修复收益。

优先跑通**小规模且可信的完整链路**，不要在训练前追求覆盖全部 29 种缺陷或扩大到不可审计的数据规模。

---

最近同步：**2026-10-12 · UniPhys-40K 首批正式 Train/Validation**；核对官方关节语义并完成40资产严格预检、5个结构候选三视图 Smoke，4个运动验证通过并形成3 Train/1 Validation catalog；未生成腐蚀或 Episode，正式 Test 与480样本 Episode 均未运行模型。
