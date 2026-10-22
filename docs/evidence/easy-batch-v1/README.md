# easy-batch-v1：Easy 批量构造（待抽查）

issue #9 的 Easy 批次：单关节物体，每个物体一个 clean 对照，加上通过自动校验的 `kinematics.joint_axis` 和/或 `kinematics.direction_flip` 样本。门槛与冻结相机协议同 `construction-pilot-v1`。没有调用任何模型。

**所有行均为 `pending_human_review`。** 本批只做抽查，抽查不等于全部人工审核；通过抽查前不进入任何基准版本，也未划分 Train/Validation。

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
