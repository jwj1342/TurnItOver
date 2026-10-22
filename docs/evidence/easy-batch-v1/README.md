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
