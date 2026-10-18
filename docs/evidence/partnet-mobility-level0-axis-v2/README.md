# PartNet-Mobility Level 0 v2

筛选 57 个同源 Train 候选物体；自动通过 28 个物体、56 个样本（clean/axis 各 28）。训练合格 28 个物体、56 个样本；待人工审核 0 个样本。目标为至少 28 个物体、56 个样本，当前状态：complete。

train_manifest.jsonl 是唯一训练输入清单；图像路径相对于本目录。private/train_targets.jsonl 由 case_id 连接监督答案，勿作为普通模型输入。object_splits.json 按独立源物体记录 Train 分组和成对资格。review_manifest.jsonl 包括待审核样本，不能直接作为训练清单。

新增待审核样本可打开 review-expansion.html 逐图查看，原生四图和 mid 六图板均可直接复核。

四图顺序：参考 rest、参考 limit、候选 rest、候选 limit；frames/ 为原生 256×256 PNG，inputs/ 为同图四宫格，boards/ 额外展示 mid。private/ 保存 Gold、程序、checker、逐关节/视角/seed 筛选诊断及旧审核记录来源。SHA 清单见 manifest.json。

新增样本仅能由真实人工逐图审核后填写 human_review.jsonl：status=human_review_complete，reviewer、reviewed_at、human_judgment.verdict、defect_id、one_glance、normal_vs_candidate_motion、evidence_states=[mid,limit]、no_clipping_or_occlusion、rest_matches_reference。请先盲审图像，不打开 private/。看不清、遮挡、出框或不明确时，status=human_review_rejected，并记录 notes；默认批准和自动筛选均不能替代逐图审核。只有同物体 clean/axis 两行均满足审核门禁才可导出。

复现（仓库根目录、全新输出路径）：

```bash
python -m scripts.replay_level0_axis_v2 --dataset docs/evidence/partnet-mobility-level0-axis-v2 --out /tmp/level0-axis-v2-repro
# 命令逐例重跑 checker 和门禁，自动核对所有重建图像 SHA。
```

人工填写审核文件后执行：

```bash
python -m scripts.finalize_level0_axis_v2 --out docs/evidence/partnet-mobility-level0-axis-v2
```

仅 human_review.jsonl 允许审核时编辑，验收先校验其余文件及源输入 SHA，再校验程序只改变一个关节的 90° 轴、checker、Train 划分、成对审核状态，最后更新 SHA。包含人工修改后，复现渲染应比较图像和程序；人工审核记录须单独保留，不能由生成器重建。
