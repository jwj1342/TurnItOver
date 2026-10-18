# PartNet-Mobility Level 0 v2 数据集

交付仍为 `docs/evidence/partnet-mobility-level0-axis-v2/`，在原有 7 个已审核独立物体上新增 21 个，合计 **28 个独立物体、56 个样本**，每个物体严格一组 clean/axis。没有建立 v3。

## 数量与审核状态

| 范围 | 独立物体 | clean | axis | 样本 |
|---|---:|---:|---:|---:|
| 原有、人工审核完成 | 7 | 7 | 7 | 14 |
| 本次新增、人工审核完成 | 21 | 21 | 21 | 42 |
| v2 合计 | 28 | 28 | 28 | 56 |
| 当前训练合格（已真实人工审核） | 28 | 28 | 28 | 56 |
| 待人工审核 | 0 | 0 | 0 | 0 |

构造与人工验收数量均已达到目标。用户在本轮明确确认：“我已逐图审核新增 42 个样本，全部通过”。据此逐行代录新增审核结论，56 行均为 `human_review_complete`，训练清单现有 56 个样本。审核者记为 user，日期为 2026-10-18（Asia/Shanghai，日期精度；用户未提供具体时刻）；运动方向描述由代理整理并注明来源，不作为用户逐字意见。原有 14 行审核记录保持不变。`validation.json` 中 `construction_target_met=true`、`acceptance=complete`、`training_eligible_cases=56`。

**[审核入口](evidence/partnet-mobility-level0-axis-v2/review-expansion.html)** 当前待审为 0。全部原始图仍保留于 `frames/`、`inputs/` 和 `boards/`，可依 `review_manifest.jsonl` 的路径逐例复核；其中四图为原生 256×256，六图板额外包含 mid。公开输入不含 Gold 或注入答案。

## 来源与独立物体划分

全部来自既有本地 PartNet-Mobility 官方 160 个 ZIP 下载池，固定 revision 为 `ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f`，没有引入其他数据源。本次从已有结构合格候选中补充物体，没有新增 ZIP 下载。源 ZIP、spec 的 SHA 均重新核验。

本次新增物体：PM_102379、PM_100174、PM_102703、PM_103008、PM_103234、PM_103319、PM_22433、PM_41003、PM_44826、PM_45213、PM_45305、PM_45523、PM_45524、PM_45573、PM_45621、PM_45699、PM_45749、PM_45908、PM_46130、PM_46230、PM_46563。

其中 PM_102379 为原 Train 物体，其他 20 个为同一来源下载池内的新增 Train 物体。28 个物体的 source model ID 和静态几何指纹均互不重复，防止仅换 archive ID 凑数量。源转换记录已有的 Test 身份/几何隔离检查均为通过。

- 原 `data/assets/partnet-mobility-trainval-v1/catalog.json` 和 `splits.json` 不修改。
- `private/expanded_splits.json` 保留原 35 个组的全部赋值，只新增 20 个 Train 组；原 Validation/Test 归属不变。
- `object_splits.json` 仅列入本次选中的 28 个物体，记录源分组、model ID、archive SHA、成对 case_id 与训练资格。
- `private/source_catalog/catalog.json` 引用包内 clean 程序的原始 spec，保存原始 spec SHA、官方来源、license、model ID、archive SHA、结构校验来源。它是复现用源目录，训练资格以实际人工审核和训练清单为准。

## Level 0 定义与采集协议

只注入单关节的 **90° `kinematics.joint_axis`** 错误；没有 part_offset、复合错误或额外源变形。验收逐字段比较原始与候选 spec，确保除了一个 joint.axis 外所有字段完全相同，axis 夹角为 90°；初始几何与参考一致。

原有 7 个物体的所有图像、程序及逐例人工判断完整保留，图像与程序 SHA 逐一核对。新增样本使用如下显式采集协议：

- rest 固定为原生状态 0；从源关节合法端点选择运动终点，mid 为从 0 到该端点的半程。关节上下限、轴、锚点及其他物体参数均不改变。
- 每个正常轨迹先计算完整几何包围范围，使用预定 padding 候选 1.25、1.00、1.50 构造固定相机；在每个相机下，参考/候选、所有状态使用完全相同的相机，不按状态重新取景。
- 只使用已有 12 个视角，按正常运动排序，最多评估前 4 个合格视角。每关节最多 24 个命中目标的确定性 axis seed，保存实际 seed、状态、相机及全部尝试结果。
- 自动图像门槛完全保留：mid/limit 参考运动及候选运动的 core changed fraction 各 ≥2%；参考/候选轨迹差异各 ≥2.5%，最大连续核心区域各 ≥750 像素；rest changed fraction ≤0.1%。仍使用原 5×5 腐蚀及 pixel delta 阈值。
- 原边缘检查继续要求所有三状态的参考和候选边缘主体占比 ≤1%。新增完整几何投影检查，所有顶点在同一固定相机内，NDC 最大绝对坐标 ≤0.98，并满足相机近远裁剪范围。
- clean 必须通过几何及预算 checker；axis 必须被几何 checker 检出，并通过预算 checker。四图全部为 Chromium 原生 **256×256**，未放大或裁剪原始渲染。

可辨识性及遮挡是否构成干扰仍必须由真实人工逐图确认；自动检查与 AI 查看不会生成“人工审核通过”记录。全部尝试及失败原因保留于 `private/expansion_screening.json`；本页只说明交付和验收，不展开拒绝原因分析。

## sft-training 读取路径

- 训练输入：`docs/evidence/partnet-mobility-level0-axis-v2/train_manifest.jsonl`。
- 私有监督：`docs/evidence/partnet-mobility-level0-axis-v2/private/train_targets.jsonl`，按 case_id 连接，仅用于 target，不能放进普通模型输入。
- 这些清单当前包含全部 56 个已真实审核的样本，clean/axis 各 28；`review_manifest.jsonl` 同步记录 56 行已完成状态。
- `images` 路径相对数据包根目录，顺序为参考 rest、参考 limit、候选 rest、候选 limit，每张 256×256。`input_preview` 为四宫格；`board` 为附带 mid 的审核图板。

审核填写 `human_review.jsonl`：审核者、日期/时间与时区、status，以及 verdict、defect_id、one_glance、运动方向描述、mid/limit 证据、rest 一致性、无出框/遮挡确认。只知道日期时明确记录日期精度，不补造具体时刻。无法确认时记录 `human_review_rejected` 及原因。同物体两行均通过真实审核才导出该组。

```bash
python -m scripts.finalize_level0_axis_v2 --out docs/evidence/partnet-mobility-level0-axis-v2
```

该命令先核验其他输出和源输入 SHA，再检查单一 90° axis、checker、原划分守恒及成对审核资格，最后更新训练清单、审核页、划分和 SHA 清单。

## 复现与验证

发布集以冻结的程序、seed、状态和固定相机为准，可在新目录直接重建所有图像，并逐例重跑 checker 与严格图像门禁：

```bash
python -m scripts.check_browser
python -m scripts.replay_level0_axis_v2 \
  --dataset docs/evidence/partnet-mobility-level0-axis-v2 \
  --out /tmp/level0-v2-replayed
```

输出路径必须不存在。命令自动逐文件比较源图像 SHA；重建不修改数据集或人工记录。`verification.json` 记录结果。新增搜索入口为 `scripts/extend_level0_axis_v2.py`，支持中断后沿检查点继续；原 7 物体快照保存在忽略目录 `data/level0-axis-v2-base-36d32f9`，亦可从本地提交 `36d32f9` 恢复。重新搜索候选与重放固定发布集是不同操作；精确图像复现请使用上述 replay 命令。

本次验证（2026-10-18）：

- 全量单元测试：283 passed、2 skipped；新增/修改的审核流程回归测试另行 7 passed。
- 相关浏览器测试：9 passed，包括全部 28 组 native 像素与 checker 重放，以及新增 21 个源物体的独立 URDF FK 对比。
- 冻结发布集重放：28 个物体、56 个样本、336 张图像全部重建，逐文件 SHA 相同，checker 与严格门禁重跑通过。
- 原 7 个物体的原生图像、程序 SHA 和人工判断保持不变；原 Train/Validation 划分逐项不变。
- 验证原始输出及机器结果存于 `private/acceptance_checks.json`、`private/replay_verification.json`。后者对应的重放源 manifest 快照为 `private/replay_source_manifest.json`。

```bash
python -m pytest -o addopts= tests/unit -q
python -m pytest -o addopts= tests/browser/test_level0_axis_v2.py \
  tests/browser/test_harness_roundtrip.py tests/browser/test_urdf.py \
  tests/browser/test_generate_smoke.py -m browser -q
git diff --check
```

没有运行 SFT、修改模型训练代码或调用付费模型；没有推送。

人工审核确认后的最终验收：审核流程回归测试 7 passed（46.44 秒）；训练输入与私有监督各 56 行，case_id 全量对应；28 个物体的训练资格均为合格，待审核为 0。全部 336 张图像与所有候选程序的 SHA 保持不变，整包文件及源输入 SHA 校验通过。
