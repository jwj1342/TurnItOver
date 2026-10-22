# Benchmark v1：数据集与评测协议（里程碑 1）

关联 [issue #9](https://github.com/jwj1342/TurnItOver/issues/9)。本里程碑只冻结数据、划分、评分和审计，**没有运行任何模型实验**。
冻结产物在 `benchmark/v1/`，实现在 `turnitover/benchmark/`，入口为 `scripts/benchmark_v1.py`。

## 难度定义

难度由物体复杂度（可动关节数）、错误类型和实际观察难度决定；复合错误数指同一样本中独立注入的错误个数。每级都含 Clean 对照（同一物体、无腐蚀、金标准 `pass`）。

| 难度 | 物体 | 缺陷类型 | 错误数 | 可见性条件 |
|---|---|---|---:|---|
| Easy | 单关节 | `kinematics.joint_axis` | 1 | 目标关节在冻结相机下 rest→limit 明显运动；参考与候选在 limit 差异通过 Level 0 像素门槛，并经人工确认无遮挡、无出框 |
| Medium | ≥2 个关节 | `kinematics.joint_axis`；单关节物体上的 `structure.part_offset`（`static_visible`） | 1 | joint_axis 沿用 Easy 的可见性门槛 |
| Hard | ≥2 个关节 | `kinematics.joint_axis`、`structure.part_offset` | 1 | 目标部件被遮挡或默认视角运动低于像素门槛，需要默认四图之外的视角或状态；与 Medium 重叠的类型按实测观察难度区分 |
| Expert | ≥2 个关节 | `kinematics.joint_axis` + `structure.part_offset` | 2 | 两个错误各自有独立的观测证据 |

当前 Level 0 样本只注入单个 90° 关节轴错误，因此能落入的只有 Easy（单关节物体）和 Medium（多关节物体上的明显轴错误）。`manifest` 对全部 60 个物体保留 Easy 候选池身份，如需把多关节物体仍并入 Easy，只需改 `difficulty_of`，不会影响划分规则。

## 数据冻结与划分

数据源：`docs/evidence/partnet-mobility-level0-axis-v4`（60 个独立物体、120 个样本，clean/axis 各 60，全部 `human_review_complete`）。

- 划分单位为独立源物体，同一物体的 clean/axis 样本必在同一侧。
- 规则：每个难度内取 `round(0.2 × 物体数)` 个 Validation 物体，仅从**未被先前实验使用**的物体中、按 `sha256("turnitover-benchmark-v1-split:" + asset_id)` 升序选取。没有任何模型输出参与划分。
- 之前的 28 个物体（`level0-axis-v2`）标记 `exposed_to_prior_experiments`，只能进 Train，不进入 Validation。
- 冻结内容：物体、`joint_id` 与 rest/mid/limit 状态值、四张 256×256 图像及其 SHA-256、候选/参考 spec 的 SHA-256、程序 SHA、来源模型 ID 与归档 SHA。`benchmark/v1/FREEZE.json` 保存四个冻结文件的 SHA-256，任何读取都会校验。

| 难度 | 物体 | Clean | Joint-axis | Part-offset | Compound | 总计 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 23 | 23 | 23 | — | — | 46 |
| Medium | 37 | 37 | 37 | 0 | — | 74 |
| Hard | 0 | 0 | 0 | 0 | — | 0 |
| Expert | 0 | 0 | — | — | 0 | 0 |
| 合计 | 60 | 60 | 60 | 0 | 0 | 120 |

| 划分 | 物体 | 样本 | Easy / Medium 物体 |
|---|---:|---:|---|
| Train | 48 | 96 | 18 / 30 |
| Validation | 12 | 24 | 5 / 7 |

## 输出格式

每个样本一行 JSON，字段与取值被严格校验，任何偏差计为 Invalid Output，不做修复或猜测：

```json
{"case_id": "l0-…", "decision": "pass|fail|uncertain|continue",
 "findings": [{"defect_id": "kinematics.joint_axis", "joint_id": "joint_0", "parts": [], "evidence": [1, 3]}]}
```

`fail` 必须带至少一个 finding，其余决策必须为空；`defect_id` 必须在 taxonomy 中；`kinematics.*` 必须给出样本上下文中的 `joint_id`，其他缺陷必须为 `null`；`evidence` 为所用输入的索引（RGB 为 0–3：参考 rest / 参考 limit / 候选 rest / 候选 limit）。`uncertain` 与 `continue` 都表示证据不足、不下结论；评分时同为弃权。

## 评分协议

| 指标 | 定义 |
|---|---|
| Macro-F1 | 对 {clean} ∪ 金标准或预测中出现的缺陷 ID 逐类 F1 取平均；pass 预测 clean，fail 预测其缺陷 ID，弃权/无效不预测任何类（幻觉的缺陷 ID 以 F1=0 计入） |
| Localization F1 | 全样本汇总的 micro F1；定位键为 (缺陷, 关节)（kinematics）或 (缺陷, 部件集)（其他） |
| Exact Verdict | 决策与预期一致且 finding 定位键集合与金标准相等；证据不足时预期为弃权 |
| Evidence Consistency | fail 预测中所有 finding 都引用合法证据、且运动学结论同时引用运动对（RGB 的 1 和 3）的比例 |
| False Pass | 在金标准有缺陷的样本上被判 pass 的比例 |
| Invalid Output | 缺失或无效预测占全部样本的比例 |

评分不含时钟与随机数；相同输入的评分文件逐字节相同。输出文件已存在则拒绝覆盖。

## 基线输入边界

边界由 `protocol.json` 的 `conditions` 固定，`condition_inputs()` 只返回允许字段，审计会检查任何条件都不读取金标准字段。

| 基线 | 可读字段 | 证据是否充分 |
|---|---|---|
| RGB | prompt、关节/部件上下文、四张冻结图像 | 是 |
| 无图像 | prompt、关节/部件上下文 | 否：正确输出是 `uncertain`/`continue`，任何 pass/fail 都是无依据的结论 |
| 像素差分 | 同 RGB 的四张图像；确定性规则，阈值只在 Train 上拟合并先冻结 | 是 |
| 代码读取 | 参考与候选 AssetSpec JSON（索引 0、1），不含图像 | 是 |

“代码”指 AssetSpec JSON：每个样本的 TypeScript 程序由 spec 派生，没有按样本落盘。

## 使用

```bash
python -m scripts.benchmark_v1 freeze          # 已冻结，重复执行会拒绝覆盖
python -m scripts.benchmark_v1 audit           # 写 benchmark/v1/audit.json；失败返回非零
python -m scripts.benchmark_v1 score --predictions preds.jsonl --condition rgb --out out/score.json
pytest tests/unit/test_benchmark_v1.py
```

审计验证：loader 读到 120 个真实样本（480 张图像全部解码并核对 SHA）；Train/Validation 的物体、`split_group`、源模型 ID、归档 SHA、几何指纹、参考程序 SHA 和图像 SHA 均无交集；Validation 不含已被先前实验使用的物体；oracle 与全弃权预测的评分重跑逐字节一致。

## 缺口（如实报告）

- **Hard、Expert 没有样本，后期再采集**；当前只采集 Easy 和 Medium。`structure.part_offset` 与复合错误同样暂无样本，Table 1 对应列为 0，需要新的构造流程。
- **Medium 只有 joint_axis**，没有“单关节物体 + 初始位置偏移”。
- **没有金标准为 `uncertain` 的样本**；弃权只能通过“无图像”条件检验。
- **Validation 只有 12 个物体、24 个样本**（Easy 10、Medium 14），指标置信区间会很宽。
- **原 Validation 的 7 个物体组**（PartNet-Mobility 1280/10797/34610/45759/100202/102252/102590）没有 Level 0 样本，不属于 v1。
- **相机视角 ID 只有 10/120 个样本有记录**；其余由图像 SHA-256 冻结相机，无法再还原视角名称。
- **120 条人工审核中 64 条是用户的整体确认**（`user_reported_visual_review_aggregate_confirmation`），不是逐例标签；其余 56 条来源记录没有标明审核方式。
- 之前 28 个物体的实验结果不得用于正式评测，这些物体只在 Train。

Validation 一经冻结不得依据模型结果修改；需要变动时创建新版本。
