# construction-pilot-v1

issue #9 的构造试点。没有调用任何模型。

## 审核结果（2026-10-22，用户盲审后对照 gold）

| 结果 | 数量 |
|---|---:|
| 缺陷样本判 fail 且与 gold 一致 | 14 |
| clean 判 pass 且与 gold 一致 | 15（用户原话“其他：pass”，为统一结论，非逐例描述） |
| 缺陷样本察觉差异但无法确定错误类型（证据不足，`human_review_rejected`） | 1：`cp-5bbf87afe780`（单关节 direction_flip，物体 PM_4542） |

- 14 个 fail 的描述与实际注入的缺陷类型全部对应：方向反向 9 个、初始位置偏移 5 个。
- **一眼可见性：** medium direction_flip 5/5 一眼可见；easy direction_flip 只有 2/5，其余 3 个（含上面证据不足的 1 个）要靠比较座位等部件的明暗；part_offset 一眼可见 2/5，其余需要多看一眼。
- 用户对两个 direction_flip 样本（`cp-51d2949ef1cd`、`cp-c30891b25542`）指出：单看候选在物理上合理，只有与参考对比才不符。这说明该缺陷只能由“参考运动一致性”判断，见 `docs/benchmark-v1.md`。
- 尚未填写：`no_clipping_or_occlusion`、`flagged_part_is_the_wrong_one`（用户未逐例提供，保持 null，不替用户补）。
- 配对规则：同物体 clean 与缺陷两行都通过才可导出，因此 PM_4542 这一对暂不合格，合格对为 14 个物体。

以下为审核前的原始说明。

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

## 换视角可见性（事后审计，2026-10-22）

`python -m scripts.audit_pilot_view_visibility` 在同一冻结相机和状态下，对 12 个视角逐个测量缺陷差异，结果见 `private/view_visibility.json`（门槛同 Level 0：核心变化 ≥2.5%，最大区域 ≥750 px）。

- 用户“靠明暗才看出”的两个 direction_flip（`cp-161d44a249bb`、`cp-c767f9b60ae3`）：默认视角 back 只有约 0.031，换到 right 约 0.072–0.077，共 9 个视角通过门槛。
- 默认视角只有一个视角通过门槛的 part_offset：`cp-43f9e326b678`、`cp-8c9e38d11cee`，数值紧贴门槛（0.025–0.026），换视角也没有更清楚。
- 像素指标与人工“一眼可见”相关性很差：`cp-5bbf87afe780` 差异很大（0.17，10 个视角通过）但用户无法判断错误类型；`cp-59278cdaa038` 默认只有 0.028 却被一眼看出。因此不能用像素门槛当作可见性或可诊断性的判据。
