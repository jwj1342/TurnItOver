# construction-pilot-v1：待人工审核

issue #9 的构造试点，**全部 30 个样本均为 `pending_human_review`，自动校验不等于人工审核**。没有调用任何模型。

15 个来自第 3 轮下载的新物体，每个物体一个 clean 对照和一个缺陷样本（同一冻结相机、同一关节状态）。

| 构造配置 | 难度 | 缺陷 | 物体 | 自动通过 / 尝试物体 |
|---|---|---|---:|---|
| easy_direction_flip | Easy（单关节） | `kinematics.direction_flip` | 5 | 5 / 24 |
| medium_direction_flip | Medium（多关节） | `kinematics.direction_flip` | 5 | 5 / 18 |
| medium_part_offset | Medium（单关节） | `structure.part_offset` | 5 | 5 / 6 |

## 怎么审

1. 打开 `review.html`（盲审：只有随机编号和图板，不显示配置或标签），**先不要打开 `private/`**。
2. 每张图板上排是参考、下排是候选，列为 rest / mid / limit，同一固定相机。
3. 在 `human_review.jsonl` 里填：`verdict`（候选对 `pass` / 错 `fail`）、`one_glance`（一眼是否明显）、`what_is_wrong`（一句话：运动方向或位置哪里不对）、`no_clipping_or_occlusion`、`flagged_part_is_the_wrong_one`（你认为出错的部件是否就是被标的那个），看不清就记 `human_review_rejected` 并写 `notes`。
4. 审完再对照 `private/gold.jsonl`。

## 重点看什么

- **direction_flip：** 候选的运动方向是否和参考相反，而且在图上看得出来。注意有些物体的运动本身对称，反向后差别很小，这类要记为不通过。
- **part_offset：** 偏移幅度固定为参考包围盒对角线的 12%，作用在可动部件上；要看 rest 列是否一眼能看出位置不对，被偏移的部件是不是你会责怪的那个。
- **clean 对照：** 候选与参考应完全一致，应判 pass。

## 自动校验（仅供参考）

冻结相机和状态只由 clean 的正常轨迹决定，缺陷渲染后不换相机；候选须通过像素差异门槛、不出框、独立 checker 判错且指向目标部件；源物体经 URDF 独立审计。所有尝试和拒绝原因在 `private/screening.json`。复现：`python -m scripts.construction_pilot --out <新路径>`。

## 已知局限

- 物体类别只有 10 种，Table 占 3 个。
- direction_flip 的通过率只有 21%–28%，主要因为运动太小或反向后差别不足。
- 本试点不是 benchmark 数据：未划分 Train/Validation，也未进入 `benchmark/v1`。
