# PartNet-Mobility Level 0 v2 数据构造与验收

交付目录：`docs/evidence/partnet-mobility-level0-axis-v2/`。本次只构造和验收数据，没有模型调用、SFT 或训练代码修改，没有改变原 Train/Validation/Test 划分。

## 实际数量与完成边界

| 阶段 | 独立源物体 | clean | axis | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 现有 Train 搜索范围 | 28 | — | — | — |
| 自动门禁通过，包含沿用 v1 | 7 | 7 | 7 | 14 |
| 沿用 v1 的真实人工审核记录，训练合格 | 5 | 5 | 5 | 10 |
| 新增，用户确认已完成人工审核，训练合格 | 2 | 2 | 2 | 4 |
| 合计训练合格 | 7 | 7 | 7 | 14 |
| 待人工审核 | 0 | 0 | 0 | 0 |
| 自动拒绝 | 21 | 0 | 0 | 0 |

**没有达到 20 个物体、40 个训练合格样本的目标。** 未放宽阈值，没有复制物体凑数。新增 PM_102314、PM_8867 的 4 个样本现已由用户确认：“我已人工审查，都没问题；可以提交commit”。据此逐行代录审核通过，当前训练合格为 7 个独立物体、14 个样本，无待审核行。审核者记为 user，日期为 2026-10-18（Asia/Shanghai，日期精度；用户未提供具体时刻），运动描述由代理整理并单独注明来源，未冒充用户逐字意见。

21 个拒绝物体：PM_100054、PM_100934、PM_101714、PM_101870、PM_102080、PM_102257、PM_102288、PM_102292、PM_102379、PM_102569、PM_102961、PM_103275、PM_10902、PM_11103、PM_1288、PM_14306、PM_15084、PM_19836、PM_25308、PM_3971、PM_48686。它们在本次搜索中所有关节、现有视角的正常运动最低 core changed fraction 均不足 0.020。因此无需注入错误，即被正常运动预筛拒绝。

这是**本次固定采集协议的筛选结果**，不是证明这些资产在其他协议下永远不可用。逐关节视角得分、所有实际渲染的 seed/view 尝试及拒绝原因在 `private/screening.json`；v1 的原始筛选记录在 `private/inherited_screening.json`。通过物体的失败尝试也保留，没有只保留成功结果。

## 沿用的质量门禁

复用 `scripts/prepare_level0_axis_review.py` 的 renderer、90° corruption、normal framing、board、四宫格、checker 和 screen，不修改 v1。每个独立源物体只保留一组 clean/axis；新物体依 catalog 稳定顺序，依次尝试其关节，使用已有视角，挑选第一个存在合格候选的关节。每个关节尝试 24 个命中目标的确定性 seed，每个已有视角均须先通过正常运动门禁，最终按原有 score 确定候选。没有随机扩大搜索直到凑够目标。

- 只改变单个关节的 axis，角度精确 90°；验收直接比较原始和候选 spec，确认其他所有字段完全相同。初始状态与参考匹配，rest changed fraction ≤ 0.001。
- 参考/候选使用同一 frozen normal framing、同视角、同 rest/mid/limit 关节值，其他关节复位；正常轨迹确定相机，不根据错误重新取景。所有 tile 均原生 Chromium 256×256。
- mid 和 limit 的参考运动、候选运动 core changed fraction 各 ≥ 0.020；参考/候选差异各 ≥ 0.025，最大连续 core 区域各 ≥ 750 像素。
- 所有 rest/mid/limit 的 reference/candidate 边缘主体占比 ≤ 0.01。该自动启发式检查不能代替人工出框/遮挡判断。
- clean 必须通过几何及运行预算 checker；axis 必须被几何 checker 检出，且通过运行预算 checker。checker 为私有特权证据，不能作为普通模型输入。
- 不使用 part_offset、复合错误、新数据源、Validation/Test 物体。

## 人工审核与训练接口

先看 `boards/` 的 rest/mid/limit 与 `inputs/` 原始四宫格，保持 native tile 大小，不打开 `private/` 查看答案。后续对新增行填写 `human_review.jsonl` 的 reviewer、reviewed_at（附时区；日期精度另行标注）、status 及 human_judgment；模板已含所需字段。逐图记录 verdict、defect_id、one_glance、正常/候选运动方向描述、mid/limit 证据、rest 是否匹配、是否无出框和遮挡。不明确时填写 `human_review_rejected` 并在 notes 解释，不能默认合格。

同一物体的 clean/axis 必须**同时**通过真实人工审核才进入训练清单。v1 原人工记录逐字沿用，来源文件和 manifest SHA 单独存放于 `private/inherited_review_provenance.json`；不冒填旧记录缺失的审核者和时间。新审核必须填写审核者、日期或时间及视觉确认字段；只知道日期时明确记录日期精度和时区，不补造具体时刻。任何程序都无法证明某个填写人真实看过图片；真实性仍依赖实际人工执行，工具只检查完整性与一致性。

`sft-training` 可直接读取：

- 输入：`docs/evidence/partnet-mobility-level0-axis-v2/train_manifest.jsonl`。
- 监督：`docs/evidence/partnet-mobility-level0-axis-v2/private/train_targets.jsonl`，按 case_id 连接。监督答案只能用于 label/target，不可拼入用户输入。
- `images` 路径相对交付目录，顺序是参考 rest、参考 limit、候选 rest、候选 limit；每张是原生 256×256 PNG。`input_preview` 是对应的四宫格。
- `object_splits.json` 以独立源物体为分组单位，全属于原 Train；每组记录成对 case_id 与训练资格。此任务不另造 Validation/Test 划分。
- `review_manifest.jsonl` 是全部候选的盲审输入；当前全部审核完成，但后续可以包含待审或被拒绝的行，训练资格始终以 `train_manifest.jsonl` 为准。

## 复现、验收与测试

在仓库根目录，使用已安装依赖及已构建的 Chromium harness；首先检查浏览器兼容性：

```bash
python -m scripts.check_browser
python -m scripts.prepare_level0_axis_v2 --out /tmp/level0-axis-v2-repro
# 输出路径必须不存在；先核对重建图像和程序，再沿用真实人工审核记录。
cp docs/evidence/partnet-mobility-level0-axis-v2/human_review.jsonl /tmp/level0-axis-v2-repro/human_review.jsonl
python -m scripts.finalize_level0_axis_v2 --out /tmp/level0-axis-v2-repro
diff -rq docs/evidence/partnet-mobility-level0-axis-v2 /tmp/level0-axis-v2-repro
```

默认输入为原 supervision config、Train/Validation splits 与 v1 bundle；生成器对源输入及 v1 文件做 SHA 检查，读取 catalog 自带的源 spec SHA。输出 manifest 固定源输入路径/SHA，逐文件 SHA 覆盖图像、程序、审核、checker、划分和训练清单。跨浏览器/渲染器版本可能产生像素差异；本机复现和浏览器重放结果另见同目录验证报告。

人工审核后只编辑 `human_review.jsonl`，然后运行：

```bash
python -m scripts.finalize_level0_axis_v2 --out docs/evidence/partnet-mobility-level0-axis-v2
```

该命令先验证其他所有输出和源输入 SHA，再检查单一 90° axis、checker、Train 分组、审核字段和成对资格，导出训练清单并更新所有 SHA。已有人类修改的 bundle 不应与全新生成的待审状态作整包字节比较；应分别比较原始图像/程序与保留的人工记录。

```bash
python -m pytest tests/unit -q
python -m pytest tests/browser/test_level0_axis_v2.py tests/browser/test_harness_roundtrip.py tests/browser/test_urdf.py tests/browser/test_generate_smoke.py -m browser -q
git diff --check
```

新增单元测试检查 SHA、原划分、Gold 隔离、审核资格门禁、图像无损拆分、文件篡改拒绝和验收幂等性。新增浏览器测试重放全部 7 组参考/候选，重新运行 checker 与严格 screen，并逐像素比较四图。既有 easy-object 测试只修正已过时的“所有记录必须未审核”断言，转为检查待审/已审记录状态与判断一致；没有改动旧数据。

## 本次验证结果（2026-10-18）

- `python -m scripts.check_browser`：Chromium 151.0.7922.34，ANGLE / SwiftShader，原生 256×256 渲染成功。
- 全量单元测试：`python -m pytest -o addopts= tests/unit -q`，281 passed、2 skipped，29.33 秒，无失败。
- 相关浏览器测试：`python -m pytest -o addopts= tests/browser/test_level0_axis_v2.py tests/browser/test_harness_roundtrip.py tests/browser/test_urdf.py tests/browser/test_generate_smoke.py -m browser -q`，8 passed，32.70 秒，无跳过。
- `python -m scripts.finalize_level0_axis_v2 --out docs/evidence/partnet-mobility-level0-axis-v2`：通过；7 个训练合格物体、14 个样本，0 行待人工审核，数量目标未达成。
- 独立构造两轮的 98 个图像及程序文件逐字节相同；SHA 校验覆盖交付包内 110 个文件。
- Python 编译与 `git diff --check` 通过。

原 v1、catalog、划分和模型训练代码均未修改；只提交本任务文件，不推送。

人工审核更新前，源文件版本另行从头构造至 `/tmp/level0-axis-v2-byte-repro-20261018`，与当时待审交付目录执行 `diff -rq`，退出码为 0：包括 manifest、人工审核模板、全部图像、程序、筛选诊断和训练导出的完整目录逐字节一致。

用户人工审核确认后，沿用独立重建目录的图像、程序和筛选记录，仅复制真实 `human_review.jsonl` 并重新验收：7 个物体、14 个合格样本，包含 manifest 在内的 111 个文件与当前交付目录逐字节相同。图板页眉保留生成时的待审状态快照；当前审核状态以 `human_review.jsonl`、`validation.json` 与训练清单为准。
