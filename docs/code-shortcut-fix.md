# 代码层捷径修复（issue #9，修复 #8 第 9 条）

**状态：** #8 第 9 条点名的捷径（候选轴不沿坐标轴、diff 参考与候选代码）已消除，`sign=-1` 同理。下文审计中拟合先验规则的 0.76（置换层）和 0.70（`direction_flip` 层）是额外审计发现的残余，不是 #8 的要求：“修复后接近随机”和 0.65 阈值是本任务的验收标准，#8 原文只要求加入读代码对照，并在其接近满分时处理。残余留到后续层处理，不算已修复；`easy-batch-v2` 与修复后的 `direction_flip` 用于 `code_read` 评测时，须同时报告这两个数字。人工审核已完成（见两个批次的 README）。

只读候选代码就能判断有没有缺陷，会让“模型是否看懂了运动”的评测失去意义。本文记录三处捷径的修复做法、审计数字和已知局限。`benchmark/v1` 已冻结，未做任何修改；所有改动都落在新协议草案（`benchmark/v2/protocol.json`）和新批次（`easy-batch-v2`）里。没有调用任何模型。

## 三处捷径与修复

### 1. `code_read` 对照可以靠 diff 满分

v1 的 `code_read` 同时给参考和候选 AssetSpec：clean 样本两份相同，有缺陷时不同，做 diff 就能满分，什么也测不出。

v2 协议（`turnitover/benchmark/protocol.py` 的 `CONDITIONS_V2`，写入 `benchmark/v2/protocol.json`，由 `python -m scripts.benchmark_v1 protocol-v2` 生成）：

- 新主条件 `code_read`：候选程序 TypeScript（`manifest.candidate_program_source`，即 `turnitover.assets.emit.emit_program` 的输出）加参考的 rest / limit 两张图，不给参考代码，也不给候选图像和 AssetSpec JSON。
- `evidence` 索引按条件定义：0 = 参考 rest，1 = 参考 limit，2 = 候选代码；运动证据对为 (1, 2)。
- 旧的双代码条件改名 `code_diff`，只作诊断，注明预期接近满分。
- `condition_inputs(row, condition, CONDITIONS_V2)` 只返回 `fields` 列出的字段，并按 `evidence` 把 `images`、`code` 裁到恰好所列角色；不会返回参考代码或任何 gold 字段（`tests/unit/test_code_shortcuts.py` 覆盖）。
- v2 只有协议草案：没有 manifest、gold，也没有 `FREEZE.json`。留出视角、25%/75% 关节值、Test 划分和 `given_joint` 条件留到冻结 v2 时再做。

### 2. `joint_axis` 的候选轴几乎都不沿坐标轴

原做法绕随机垂直方向旋转 90°：v1 为 60/60、`easy-batch-v1` 为 45/47 不沿坐标轴，而 clean 轴绝大多数沿坐标轴（round3 为 573/590）。

`kinematics.joint_axis` 新增 `mode`（缺陷 id 不变，`params.mode` 记录）：

- `axis_permutation`：把 `±e_i` 换成另一条坐标轴 `±e_j`（j≠i），在 4 个目标中用显式的 `numpy.random.Generator` 抽取，正负号也由它抽。参考轴不沿坐标轴的关节不参与（`eligible` / `applicable`）；批次构造时整个物体被跳过并写入 `screening.jsonl` 的 `reason`（本批没有这样的物体）。
- `axis_random_angle`：原行为，保留为单独的一层。gold 里 `stratum` 区分两层。

### 3. `direction_flip` 的 `sign=-1`

`easy-batch-v1` 的 29 个缺陷全是 `sign=-1`，50 个 clean 全是 `sign=1`。

改为把 `axis` 取负、`sign` 保持不变（`params.mode = axis_negation`）。两种写法的运动完全相同（单元测试对旋转和平移关节在整个限位范围内逐点验证）；clean 的轴本来就有正有负，所以 `sign` 不再露出缺陷。取负用 `0.0 - x`，避免生成代码里出现 `-0.0`。

## `easy-batch-v1` 的处理

- **29 个 `direction_flip`：原地修复。** 只重新生成候选 spec，然后在冻结相机下重新渲染，逐张核对 4 张图的 SHA-256：**29/29 与已提交一致**（明细 `docs/evidence/easy-batch-v1/direction_flip_regeneration.json`），所以图像和人工审核继续有效。命令 `python -m scripts.rebuild_easy_batch flip`。
- **47 个 `joint_axis`：作废**（gold 的 `training_eligibility = superseded_code_shortcut`），用置换模式重新构造为 `easy-batch-v2`：50 个物体中 **42 个通过**同样的像素门槛，8 个四个候选均未通过（其中 5 个在 v1 有随机角度样本：PM_100685、PM_10305、PM_10357、PM_11581、PM_9992）。全部 `pending_human_review`。命令 `python -m scripts.rebuild_easy_batch axis`。
- 第一次抽查里 4 个 `joint_axis` 的结论保留为历史记录，不作为有效样本的审核依据。clean 与 `direction_flip` 的审核照常有效。详见两个批次的 README。

## 审计

`python -m scripts.audit_code_shortcuts all`（只读；代码 `turnitover/benchmark/code_shortcuts.py`，输出 `docs/evidence/code-shortcut-audit-v1/`，其中 `features.json` 保存每个关节的轴、sign、锚点等特征，复现不再需要大 spec 文件）。

只看候选代码，不用参考。指标是平衡准确率（缺陷召回与 clean 召回的均值，随机为 0.5），括号内为 95% 自助法区间。固定规则事先给定，按其较好的方向报告（等于攻击者用少量标注翻转规则）；拟合规则用留一法从其余样本学习类条件频率，不翻转。规则：

| 规则 | 含义 |
|---|---|
| `axis_off_coordinate` | 有关节的轴不沿坐标轴 → 缺陷 |
| `sign_negative` | 有关节 `sign<0` → 缺陷 |
| `axis_dominant_negative` | 主分量为负的轴 → 缺陷 |
| `origin_tidy` | 锚点和部件原点都是整齐数值（3 位小数内）→ clean |
| `axis_signature_prior`（拟合） | 按“轴落在 ±x/±y/±z 或不在坐标轴”的类条件频率 |
| `axis_index_prior`（拟合） | 同上，只看 x/y/z 下标，不看正负 |

每个 stratum 的 clean 取该数据集的全部 clean。数据集：`benchmark-v1`（60 缺陷/60 clean）、`easy-batch-v1` 修复前（直接读保留的 `easy-batch-v1-specs-prefix`）、修复后（clean 与修复后的 29 个 `direction_flip` 来自 `easy-batch-v1`，42 个置换样本来自 `easy-batch-v2`）。

| 数据 | 缺陷 / stratum | n 缺陷/clean | 最佳规则 | 最佳 | 其他规则 |
|---|---|---|---|---:|---|
| benchmark-v1 | joint_axis / axis_random_angle | 60/60 | axis_off_coordinate | **0.97** (0.93–0.99) | index 0.95, signature 0.91, dominant_neg 0.54, origin_tidy 0.50, sign 0.50 |
| easy-batch-v1 修复前 | joint_axis / axis_random_angle | 47/50 | axis_off_coordinate | **1.00** | index 1.00, signature 1.00, dominant_neg 0.51, origin_tidy 0.50, sign 0.50 |
| easy-batch-v1 修复前 | direction_flip / sign_negation | 29/50 | sign_negative | **1.00** | origin_tidy 0.57, dominant_neg 0.53, signature 0.51, index 0.48, off_coordinate 0.50 |
| 修复后 | joint_axis / axis_permutation（v2） | 42/50 | axis_signature_prior（并列 index） | **0.76** (0.67–0.83) | dominant_neg 0.55, origin_tidy 0.51, off_coordinate 0.50, sign 0.50 |
| 修复后 | direction_flip / axis_negation | 29/50 | axis_signature_prior | **0.70** (0.60–0.80) | origin_tidy 0.57, dominant_neg 0.55, index 0.48, off_coordinate 0.50, sign 0.50 |

### 结论：两种明显的捷径消除了，但没有达到“接近随机”

- 被点名的三条规则已经失效：`axis_off_coordinate` 与 `sign_negative` 在修复后都是 0.50（置换样本与取负样本的轴都沿坐标轴，`sign` 全为 1）；`origin_tidy` 一直在 0.5–0.57，不是捷径。
- **但两层仍明显高于 0.65，不能说修复完成：**
  - **`axis_permutation` 0.76**：clean 轴以 x 为主（x+ 11、x− 23、y+ 6、y− 3、z+ 7；没有 z−），置换后的候选轴多在 y/z（x+ 1、x− 6、y+ 7、y− 13、z+ 8、z− 7）。“轴是 x → clean，轴是 y/z 或 −z → 缺陷”就能得到约 0.76。原因是目标轴在 j≠i 里均匀抽取，而 clean 的轴下标分布并不均匀，且 x 轴居多的物体更容易被换成非 x。
  - **`direction_flip` 0.70**：取负把 x− 的 clean（23 个）换成 x+，使缺陷样本以 x+ 为主（x+ 16、x− 8、y+ 1、y− 4），而 clean 以 x− 为主。这是 clean 轴符号分布不均匀的后果；`axis_off_coordinate` 和 `sign_negative` 都看不出来，但按符号分布的拟合规则能看出来。
- 规则没有为了压低数字而调整；上面是首次运行的结果，规则在运行前就已写定。

### 修改方案（尚未实施，需用户确认后再做）

1. **`direction_flip`、也适用于所有轴符号：** 对每个样本（clean 和缺陷都做）用显式 `Generator` 掷硬币，把同一个运动写成 `(axis=a, sign=+1)` 或 `(axis=−a, sign=−1)`。运动不变，图像 SHA 不变；这样 clean 与缺陷的“轴符号 × sign”分布完全相同，`sign_negative` 与符号先验都回到 0.5。代价：clean 候选代码也要重新生成（`easy-batch-v1` 的 clean 图像不受影响）。
2. **`axis_permutation`：** 目标轴下标不再均匀抽取，而按同批 clean 轴下标的经验频率加权（对 j≠i 归一化），或者在冻结前对缺陷按“下标”做匹配子采样，使两类的下标分布一致；配合第 1 条后只剩下标分布。需要更大的 clean 池，匹配子采样会丢掉一部分已通过的样本。
3. 实施后用同一审计重跑，目标是所有规则的置信区间包含 0.5。用户决定推迟到后续层处理；推迟期间这两层不能被描述为“代码层无捷径”。

## 已知局限

- 审计规则是一组简单规则，不是上界：更强的拟合（例如结合几何、分段特征）可能得到更高的数字；本数据的 clean 只有 50 个物体，区间较宽（宽度约 ±0.08–0.1）。
- 审计只看轴、sign、锚点和部件原点；网格顶点、材质和代码长度没有被审计。这些在缺陷和 clean 之间由同一 emitter 生成，且 `joint_axis` / `direction_flip` 不改动它们，因此预期不构成捷径，但没有验证。
- `part_offset`（Medium）不在本次范围，它会改动部件原点，需要在实现后单独审计 `origin_tidy`。
- 缺陷样本只来自通过像素门槛的物体，门槛筛选是否改变了轴分布（例如 x 轴更易通过）没有单独评估。
- `easy-batch-v2` 的 8 个未通过物体改变了物体构成；5 个物体只剩 clean 与（至多）`direction_flip` 样本。
- `code_read` 的 v2 条件只有协议，还没有真实的 manifest 和打分路径，`scoring.py` 与 `audit.py` 仍只服务 v1；新条件的评分在冻结 v2 时接入。
- 仓库里已有 4 个与本任务无关的 pytest 失败（`test_level0_axis_*`：脚本 SHA 校验，见下），不是本次引入。
