# Qwen3-VL-2B 正式 SFT 实验记录

本文持续记录正式 Train/Validation 数据上的训练与受控对照；冻结 Part 2 Test 不用于训练、模型选择或本轮评测。

## 第一轮：PartNet-Mobility terminal diagnosis SFT

### 数据边界与预检

数据来自 `data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3`。loader 实际核验 manifest 完整且 `formal_training=true`，逐条校验图像路径及 SHA-256、来源身份/内容 SHA-256、候选与参考程序 SHA；Train 为110条 terminal Episode、28个原物体，Validation 为28条、另7个原物体，原物体和 `split_group` 交集均为0。两种监督逐样本使用相同用户消息、图像列表和终止位置；真实 Qwen processor 已成功生成多图张量及只覆盖 assistant token 的 loss mask。

训练不读取或拟合 Oracle action 行。当前没有合格 `pass`：clean 样本仍监督为 `uncertain`，没有根据隐藏的空 corruption 列表制造 `pass`。

两种 SFT 的唯一区别是 terminal assistant target：

| 设置 | Train 样本 | target finding | 定义 |
|---|---:|---:|---|
| 普通 SFT | 110 | 111 | 对72个有缺陷样本监督全部注入的最终诊断；38个 clean 保持 `uncertain`。未以可见 exposure 筛选 finding。 |
| 证据约束 SFT | 110 | 74 | 只保留公开历史中通过 counterfactual pixel-exposure 门禁的 finding；未暴露缺陷不进入目标，38个无 finding 样本为 `uncertain`。 |

普通组 target 为0/1/2 findings 的样本数是38/33/39，证据组为38/70/2。两组输入完全相同，但 target 长度不可避免不同：普通组共17,253个 assistant token，证据组13,874个。

这里的“证据”是已通过数据质量门禁并与真实渲染绑定的 counterfactual pixel proxy，不是自然错误的独立人工 Gold，也不证明模型能主动选择动作。

### 训练设置与产物

两组均使用同一 Qwen3-VL-2B-Instruct、本地 bf16、LoRA `q/k/v/o_proj`、rank 8、alpha 16、dropout 0、AdamW、学习率 `2e-4`、seed `20261013`、3 epochs、每样本一步，共330 optimizer steps。可复现参数见 [`partnet-mobility-sft-v1.json`](../configs/partnet-mobility-sft-v1.json)。两张 RTX 4090 D 并行运行；CUDA/Flash Attention 日志提示内核不是逐 bit 确定性，因此固定 seed 只控制样本顺序和随机初始化。

| 训练 | first loss | epoch mean 0 / 1 / 2 | last loss | adapter SHA-256 |
|---|---:|---|---:|---|
| 普通 SFT | 2.413 | 0.391 / 0.181 / 0.180 | 0.263 | `dd473181652e370dbe7d6e5d7a7e0852bfe0e30bb6cb0be4dfe69b77b51b227c` |
| 证据约束 SFT | 2.446 | 0.382 / 0.166 / 0.161 | 0.267 | `58ed2f0f3b246ffae5781104be11209333b5af232717315b3ccf10847b1be37f` |

完整逐 step 训练曲线位于两个运行目录的 `loss.jsonl`，配置/环境摘要在 `run.json`，checkpoint 在 `checkpoint/`。这些目录及图像均受 `.gitignore` 排除，不进入代码提交：

- `output/partnet-qwen3vl2b-ordinary-sft-v1/`
- `output/partnet-qwen3vl2b-evidence-sft-v1/`

### 独立 Validation 对照

三组使用相同28条 Validation 输入、greedy decode 和 `max_new_tokens=512`。`stop` 表示在强制 terminal prompt 上生成可解析最终 verdict，不是主动动作策略的停止准确率。定位以 `(defect_id, joint_id, parts)` 精确匹配；证据一致要求每个预测 finding 匹配审核目标，且引用步骤属于该 finding 的有效公开证据步骤。

| 指标 | 未微调 | 普通 SFT | 证据约束 SFT |
|---|---:|---:|---:|
| 合法输出 / terminal stop | 28/28 | 28/28 | 28/28 |
| status accuracy | 7/28 (25.0%) | 28/28 (100%) | 28/28 (100%) |
| exact verdict/定位 | 7/28 (25.0%) | 9/28 (32.1%) | 13/28 (46.4%) |
| finding precision / recall / F1 | n/a / 0 / 0 | 0.263 / 0.526 / 0.351 | 0.375 / 0.474 / 0.419 |
| finding TP / FP / FN | 0 / 0 / 19 | 10 / 28 / 9 | 9 / 15 / 10 |
| 证据一致输出 | 26/28* | 9/28 | 13/28 |
| unsupported / predicted findings | 0 / 0 | 33 / 38 | 15 / 24 |
| 错误 fail / 错误 pass | 0 / 2 | 0 / 0 | 0 / 0 |

`*` 底座26次 `uncertain` 没有无依据 finding，但这不能解释为诊断能力；它遗漏全部19个目标 finding，并错误输出2次 `pass`。两种 SFT 都学会了本轮的 fail/uncertain 停止分布，但证据组相对普通组减少18个 unsupported finding、将 exact verdict 提高4例并提高 precision/F1，同时少识别1个目标 finding。第一轮支持“证据筛选减少无依据定位”的描述，不支持泛化、校准或主动取证结论。

按原始 Validation 物体汇总如下；每个物体4个样本，单元格为 `status正确 / exact / unsupported findings`：

| 原始物体 | 未微调 | 普通 SFT | 证据约束 SFT |
|---|---|---|---|
| PM_100202 | 2 / 2 / 0 | 4 / 2 / 4 | 4 / 2 / 2 |
| PM_102252 | 0 / 0 / 0 | 4 / 0 / 7 | 4 / 2 / 2 |
| PM_102590 | 1 / 1 / 0 | 4 / 1 / 4 | 4 / 1 / 3 |
| PM_10797 | 1 / 1 / 0 | 4 / 3 / 1 | 4 / 3 / 1 |
| PM_1280 | 0 / 0 / 0 | 4 / 0 / 7 | 4 / 1 / 3 |
| PM_34610 | 0 / 0 / 0 | 4 / 0 / 8 | 4 / 1 / 3 |
| PM_45759 | 3 / 3 / 0 | 4 / 3 / 2 | 4 / 3 / 1 |

逐案例预测及聚合结果位于：

- `output/partnet-qwen3vl2b-base-val-v3/{predictions.jsonl,metrics.json}`
- `output/partnet-qwen3vl2b-{ordinary,evidence}-val-v2/{predictions.jsonl,metrics.json}`
- `output/partnet-qwen3vl2b-controlled-comparison-v3.json`

### 复现命令

```bash
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/train_qwen_episode_sft.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --out output/partnet-qwen3vl2b-ordinary-sft-v1 --supervision ordinary --seed 20261013 --epochs 3 --learning-rate 2e-4 --lora-rank 8 --lora-alpha 16
CUDA_VISIBLE_DEVICES=1 ../TurnItOver/.venv-qwen/bin/python scripts/train_qwen_episode_sft.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --out output/partnet-qwen3vl2b-evidence-sft-v1 --supervision evidence --seed 20261013 --epochs 3 --learning-rate 2e-4 --lora-rank 8 --lora-alpha 16

CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/evaluate_qwen_episode_sft.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --out output/partnet-qwen3vl2b-base-val-v3 --max-new-tokens 512
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/evaluate_qwen_episode_sft.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --adapter output/partnet-qwen3vl2b-ordinary-sft-v1/checkpoint --out output/partnet-qwen3vl2b-ordinary-val-v2 --max-new-tokens 512
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/evaluate_qwen_episode_sft.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --adapter output/partnet-qwen3vl2b-evidence-sft-v1/checkpoint --out output/partnet-qwen3vl2b-evidence-val-v2 --max-new-tokens 512
python scripts/compare_qwen_sft.py --base output/partnet-qwen3vl2b-base-val-v3/metrics.json --ordinary output/partnet-qwen3vl2b-ordinary-val-v2/metrics.json --evidence output/partnet-qwen3vl2b-evidence-val-v2/metrics.json --out output/partnet-qwen3vl2b-controlled-comparison-v3.json
```

## 已知限制与后续消融

Validation 只有7个原始物体、28例，且类别来自同一 PartNet-Mobility来源；当前指标不能估计跨来源泛化。数据只有 `fail/uncertain`，无法评估有证据的 `pass`。观察序列由私有教师预先取得，本轮只比较固定观察后的诊断与停止，不评价主动动作。普通组与证据组 target token 数不同，虽然样本、输入、步数、优化器和 seed 相同，token 级监督量并非完全相等。

下一轮优先进行：多 seed 置信区间；按 target token 或 optimizer token 匹配预算；将单缺陷与组合缺陷分层；加入自然错误人工 Gold；在不读取隐藏反事实的条件下训练动作/停止联合策略。完成 Validation 方案冻结前不运行 Part 2 Test。

## 2026-10-17：Motion-grounded SFT v2 已完成

两组 Qwen3-VL-2B 像素面板对照各完成216步真实GPU训练和三组评测；显式运动比较中间监督未改善冻结B上的offset（均0/4）、clean（均0/4）或完整诊断（均4/12）。停止重复2B LoRA，暂不接入v5停止/主动策略；上述早期后续计划不作为本轮继续训练依据。完整结果、审计、checkpoint路径和复现命令见[本轮报告](evidence/motion-grounded-v2-20261017/REPORT.md)。
