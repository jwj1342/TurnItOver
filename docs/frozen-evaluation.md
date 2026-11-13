# 数据更新后的指标重算

评测输出保存实际输入、金标和图片快照；离线统计不再依赖原数据目录。
每个实验输出目录单独冻结版本，不覆盖旧结果，不混用不同数据版本的分母。

## 已有 Episode 导出

`scripts/evaluate_qwen_episode_sft.py` 原命令继续有效。新增可选
`--training-seed` 用于记录 checkpoint 的训练 seed；`--seed` 仍是推理 seed，
二者不能混用。评测在模型调用前自动生成 `scoring-snapshot/`：

- 实际评测 records，包括原始提示词、target、source identity 和 evidence。
- 仅这些案例的私有 annotations；不额外复制其他 split 的金标。
- 按内容 hash 去重的原始图片，以及原路径和快照路径映射。
- 数据 manifest、源文件 hash、文件清单和实际 taxonomy。

`predictions.jsonl` 继续保存 raw、解析结果、token 和耗时；`metrics.json`
增加快照/预测文件 hash、代码版本、adapter 文件 hash 和训练 seed。
快照与预测均含私有内容，应与实验结果一起保存在私有存储中。

离线重算，无需 GPU 或原数据目录：

```bash
python scripts/score_frozen_evaluation.py \
  --evaluation /absolute/path/to/completed-run \
  --out /absolute/path/to/new-metrics.json
```

输出保留旧 target 指标，并在 `paper_metrics` 下单独给出：

- `all_injected_metrics`：以完整注入金标计算 defect Macro-F1 和 part micro-F1。
- `strict_evidence_support_rate`：完整 finding 签名正确，且引用步骤集合完全匹配。
- `exposure_any_rate` / `exposure_all_rate`：只在 defective cases 上统计。
- `unexposed_finding_rate`：展示观察未暴露该预测 defect 类型的 finding 比例。
- 各 tier 的独立统计；案例数和类别集合从快照读取，不写死为 120/五类。

完整曝光映射缺失时，相关指标为 `null` 并给出原因。旧 target 引用可作为
严格匹配的回退来源，但会明确标记其不是完整曝光审计。没有预测 finding 时，
以 finding 数为分母的比例为 `null`。非法回复保留在案例集合，产生漏检。

未完整覆盖所有案例、重复预测 ID、文件 hash 不符或未完成的运行均拒绝汇总。
已有历史实验若未保存快照，不能凭最新数据自动补造旧快照。

## 更新规则

1. 原数据更新或移除：旧快照仍可重算旧实验。
2. 新增案例、变更图片/观察/提示词：新建实验目录，对变化后的输入重新推理。
3. 修订金标：须明确建立新的金标版本和评分协议；本命令不会自动拿新金标覆盖旧结果。
4. 新增 taxonomy ID：需同步模型输出解析器支持；不支持的 schema 应先添加适配，不能静默计零。

本改动只补固定观察统计与记录，不新增训练或模型调用。Withheld 测试仍需实际
删去观察后重新推理；修复表的逐缺陷 resolved 仍需对应独立 checker 和逐轮成本账本。

## TIO merged750 原生发布包

兼容发布标签 `three-tier-rp29-merged750-complete-candidate-v1-pre.1` 解压后的完整包，
不要求先转换成三 paired SFT 导出。其原生 `manifest.private.json` 的 SHA256 为
`71f8b8d48173425f0bbecb5c1247f425ea151ef5730d58ac2f621d23c89e91c3`。

先冻结所选 split，默认 Validation，不自动展开 Test：

```bash
python scripts/snapshot_candidate_evaluation.py \
  --dataset /absolute/path/to/tio750-complete-candidate-v1 \
  --split validation \
  --manifest-sha256 71f8b8d48173425f0bbecb5c1247f425ea151ef5730d58ac2f621d23c89e91c3 \
  --out /absolute/path/to/candidate-validation-snapshot
```

`inputs.jsonl` 只含公开的案例 ID 和 reference/candidate 图片对、view/state；
图片路径相对快照目录。应将这些实际输入交给模型，并保存每例原始回复；此入口
负责记录和统计，不调用模型，也不修改现有训练 loader。原生包每例图片对数量
可变，不能截断或补齐成三对。私有 case/Gold 文件独立保存，不进入公开输入。

预测文件每行包含 `id`（或 `case_id`）和字符串 `raw`。`raw` 的 JSON 协议例如：

```json
{"verdict":{"status":"fail","findings":[{"defect_id":"kinematics.joint_axis","evidence_steps":[1]}]}}
```

这里 `evidence_steps` 为从零开始的图片对索引。外层可另存 `input_tokens`、
`output_tokens`、`usage`，评分时原样保留。所有案例都要有记录；格式错误的
`raw` 保留为 invalid，不能删掉失败案例以减少分母。

```bash
python scripts/score_candidate_evaluation.py \
  --snapshot /absolute/path/to/candidate-validation-snapshot \
  --predictions /absolute/path/to/predictions.jsonl \
  --out /absolute/path/to/new-candidate-scores
```

原始发布目录之后即使更新或删除，也不会影响该快照统计。新版本使用新的
manifest hash 和输出目录。输出包含固定完整 taxonomy Macro-F1、只对有正例
支持类别的 Macro-F1、逐类计数和逐 tier 结果；完整类别范围明确写入结果。
Validation 实际仅覆盖 10 类，不能把支持类别得分宣称为 29 类全覆盖表现。

该包未统一提供经过审核的逐观察曝光映射和部件定位金标，故 Exposure、
Evidence support、Unexposed rate 和 Localization 返回 `null` 与原因。
不从像素差、构建缺陷或缺失证据反推语义曝光。候选的
`pending_human_review`、训练/正式准入状态随快照和结果保留；统计兼容不改变准入。
