# easy-batch-v1：Easy 批量构造（已抽查 20 个，其余待审）

issue #9 的 Easy 批次：单关节物体，每个物体一个 clean 对照，加上通过自动校验的 `kinematics.joint_axis` 和/或 `kinematics.direction_flip` 样本。门槛与冻结相机协议同 `construction-pilot-v1`。没有调用任何模型。

**只有抽查的 20 个样本有人工记录，其余 106 个仍是 `pending_human_review`。** 抽查不等于全部人工审核；本批不进入任何基准版本，也未划分 Train/Validation。

## 抽查结果（2026-10-22，盲审后对照 gold）

| 结果 | 数量 |
|---|---:|
| 缺陷样本判 fail 且与 gold 一致（joint_axis 3、direction_flip 6） | 9 |
| clean 判 pass 且与 gold 一致（用户原话“其他pass”，统一结论，非逐例描述） | 10 |
| direction_flip 察觉差异但无法确定错误类型（证据不足，`human_review_rejected`） | 1：`eb-83bfe159d125`（PM_4578） |

- 9 个 fail 全部“一眼不对”，描述与注入的缺陷类型对应；`eb-ef0a282b1dc6` 是 joint_axis，用户描述为“运动方向错了”，判定正确但没有区分到轴。
- 试点和本批各出现 1 个“看得出不对、说不出错在哪”的 direction_flip，两次都是同一种缺陷。目前样本太少，不能判定为系统性问题，需要在后续批次继续观察。
- 另一会话对 direction_flip 的编码改动（见 `direction_flip_regeneration.json`）重渲染的 29 个样本图像逐字节一致，抽查结论按图像有效。

| 项目 | 数量 |
|---|---:|
| 尝试物体 | 143 |
| 自动通过物体 | 50（通过率 35%） |
| 样本 | 126：clean 50、joint_axis 47、direction_flip 29 |
| 两种缺陷都通过的物体 | 26；只通过一种的 24 |

类别：Dishwasher 7、Laptop 7、StorageFurniture 6、TrashCan 5、Display 5、Box 4，其余 10 类各 1–3 个；每类上限 8。物体与已有 v4 和试点物体无重复。

## 抽查

`spot_check.json` 按固定哈希顺序取 10 个缺陷样本及其 clean 对照（共 20 个），打开 `review.html` 盲审，**先不要打开 `private/`**。在 `human_review.jsonl` 对应行填写，规则同 `construction-pilot-v1/README.md`。若发现系统性问题，按 issue 要求先修规则并复查受影响批次。

## 复现与存放

- 候选 spec 约 263 MB，放在 `data/benchmarks/easy-batch-v1-specs/`（不进 Git），`manifest.json` 记录每个文件的 SHA-256；图像、图板和审核页在本目录。
- `python -m scripts.construct_easy_batch --out <新路径> --specs <新路径>`，可中断后续跑。
- `private/view_visibility.jsonl` 记录每个缺陷样本在 12 个视角下的差异，供后续按“是否需要补视角”分层（`scripts/tier_pilot.py` 的规则尚未应用到本批，因为需要人工审核的“一眼可见”结果）。

## 代码层捷径修复后的状态（issue #9 / #8 第 9 条）

只读候选代码就能判断有没有缺陷，见 `docs/code-shortcut-fix.md`。本批受影响如下：

- **`direction_flip`（29 个）已原地修复。** 候选 spec 重新生成：`axis` 取负、`sign` 保持 1，运动与原先 `sign=-1` 完全相同。逐张重新渲染，4 张图的 SHA-256 与已提交的一致（29/29，明细见 `direction_flip_regeneration.json`），所以图像、图板和人工审核结论继续有效。`private/gold.jsonl` 新增 `stratum`：已修复的为 `axis_negation`。
- **`joint_axis`（47 个）作废，由 `easy-batch-v2` 取代。** 它们的随机垂直轴 90° 旋转几乎都不沿坐标轴（45/47）。`private/gold.jsonl` 中这 47 行的 `training_eligibility` 改为 `superseded_code_shortcut`，`stratum` 为 `axis_random_angle`；文件与图像保留作历史记录，不再进入任何基准或训练集。其中 42 个在 v2 有置换后的对应样本。
- **`6cd197b` 抽查中那 4 个 `joint_axis` 审核结论**（`eb-449cfe8aaca5`、`eb-cad7bbbd813e`、`eb-ce72ec9b2d36`、`eb-ef0a282b1dc6`，均为 fail）保留为历史记录，**不再作为有效样本的审核依据**；v2 的样本需要重新审核。
- **clean（50 个）不变。** 它们的 reference 图像与 v2 逐张核对一致，clean 对照继续有效，审核记录继续有效。

### 5 个在 v2 中没有对应样本的物体

PM_100685、PM_10305、PM_10357、PM_11581、PM_9992 在随机角度下通过过像素门槛，改用坐标轴置换后四个候选都没有通过（`easy-batch-v2/private/screening.jsonl`）。这些物体没有 `joint_axis` 样本，**clean 对照照常保留**（`eb-518cd17f43d2`、`eb-3951626dee9f`、`eb-92c5253d586a`、`eb-8ce90fc04986`、`eb-24a1fff115b8`）：clean 样本只依赖物体与门槛，不依赖缺陷是否构造成功，而且不混入任何缺陷内容。其中 PM_100685 仍有已修复的 `direction_flip` 样本。注意：这几个 clean 没有配对的 joint_axis 缺陷，若将来要求每个 clean 都有配对缺陷，应单独筛掉；这样做是否引入类别偏差，尚未评估。

### 抽查现状

`spot_check.json` 中原有的 4 个 `joint_axis` 样本不用审；clean 与 `direction_flip` 照常审（第一次抽查已完成 19/20 与 gold 一致、1 个证据不足，见 `human_review.jsonl`）。`joint_axis` 的新样本见 `easy-batch-v2`。

`manifest.json` 已重新生成：`private/gold.jsonl` 与 29 个 `direction_flip` 候选 spec 的哈希更新，`human_review.jsonl` 仍不在哈希之内。
