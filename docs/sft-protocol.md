# Qwen SFT 与固定观察评测协议

本 PR 从 PR #7 (`laq/uniphys-e2e`, `0242f21`) 提取训练与评测基础设施。
它依赖 #7 提供的 Episode 导出、verifier 契约和 Qwen processor adapter。
不包含数据构造、教师采集、浏览器闭环、Action SFT、联合训练、课程训练、Level 0、motion-grounded、visual stopping 或 Kimi 实验。历史研究代码和附件继续保留在 `laq/sft-training`；[历史结论](sft-experiments.md) 不作为本次代码的验证证据。

## 安装与基础测试

Python 3.12。常规测试无需 GPU、模型或私有数据：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest tests/unit
```

Torch 是 loss-mask 测试的可选依赖，缺失时明确 skip。真实 Qwen 训练/推理另外安装：

```bash
python -m pip install -r requirements-qwen.txt
```

2B、4B、8B 均使用本地 Qwen3-VL Instruct 模型目录；提前按对应模型许可证下载 processor、tokenizer、config 和权重。入口不自动下载权重，不使用 API 密钥。显存需求随模型、图片数量和分辨率变化；本次未回归完整规模矩阵。

## 数据契约与审核边界

调用者必须提供已审核导出的 manifest SHA-256；摘要匹配只验证版本身份，不能代替审核。`manifest.json` 必须 `status=complete` 且 `formal_training=true`。诊断读取 `<split>.episodes.jsonl` 的 terminal 行；停止读取 `<split>.stopping.jsonl` 的因果前缀。train 只消费 Train 目标。split 审计检查现有导出文件的来源元数据，不把 Validation/Test 目标交给模型或优化器。没有导出的 split 无法据此验证；发布数据时应提供完整 split 元数据。

每行需要 `id`、`episode_id`、`images`、逐图 `image_sha256`、`data_identity` 和一对 user/assistant 消息。`data_identity` 包含 `source_dataset`、`source_asset_id`、`asset_id`、`split_group`、64 位来源身份/内容 SHA、16 位 reference/candidate 程序 SHA 和 `formal_training`。公开 payload 放在 user 消息最后一行 JSON，包含 `context`、`history`、`image_order`，图片顺序必须等于 reference 加历史实际观察。路径必须留在 dataset 根目录内；绝对路径、逃逸根目录的目录穿越及 symlink 均拒绝。图片内容逐张核对 SHA。来源身份相互矛盾、同一原物体跨 split、重复样本或公共观察与图片不一致时拒绝加载。

来源身份 SHA 必须等于 `sha256(source_dataset + ":" + source_asset_id)`。所有任务都需要 `annotations.private.jsonl`，仅用于来源/曝光标签审计；诊断按 `covered_label_indices` 核对 finding 身份，停止核对索引合法性。这些私有审核字段不进入模型输入。

公开 payload 仅接受约定字段，并递归拒绝已知教师私有键（labels、exposures、future_history 等）。这不等于任意自由文本的保密审查；prompt 前缀和观察文本仍需数据审核确认不含 Gold、源码捷径或未来信息。

这些校验能检测导出内的矛盾与内容变化，不能证明来源声明真实。真实研究数据仍需独立审核来源、曝光门禁与数据用途；不得把合成 fixture 的 `formal_training` 声明解释为真实数据已审核。

## 监督与 loss

`--task diagnosis` 和 `--task stopping` 分别运行，不混合优化。`--supervision ordinary|evidence` 使用相同公开输入、图片与顺序；区别只在诊断 assistant 目标。Ordinary 从同身份的私有注入标签构造最终诊断，不按可见曝光筛选；Evidence 校验审核曝光，仅监督已观察证据支持的 findings。Stopping 的两种模式共用同一因果前缀目标。未观察到缺陷、空隐藏标签或预算耗尽均不能单独产生 `pass`；当前导出不提供正向 pass 证据。

私有注入标签和曝光元数据只供构造/审计目标，不进入 user prompt。Evidence support 指标是与审核曝光目标及引用步骤的一致性，不能称为独立人工 grounding。Ordinary 与 Evidence 的目标长度会不同，不宣称 token 监督量相等。

训练复用 `tokenize_example`：图像、用户文本、历史观察及 padding 的 label 均为 `-100`，只对 assistant 回答计算 loss。保留每样本一步、assistant token mean loss、AdamW、梯度裁剪 1.0、LoRA q/k/v/o projection、dropout 0；CUDA 使用 bf16 与 SDPA。CPU float32 路径用于小型链路 smoke。固定 seed 控制初始化与样本顺序，CUDA warn-only 确定性设置不保证逐 bit 复现。

## 运行命令

配置模板 [partnet-mobility-sft-v1.json](../configs/partnet-mobility-sft-v1.json) 保留历史优化参数，不预填机器路径或审核 SHA。显式指定版本、模型、监督与 seed：

```bash
python scripts/train_qwen_episode_sft.py \
  --config configs/partnet-mobility-sft-v1.json \
  --dataset /path/to/audited-export --dataset-manifest-sha256 SHA256 \
  --model /path/to/Qwen3-VL-2B-Instruct --out output/diagnosis-evidence \
  --task diagnosis --supervision evidence --seed 20261013
```

换用 `--task stopping` 创建独立 checkpoint；换用 `ordinary` 创建独立对照。不要使用同一输出目录覆盖既有结果。训练目录包含 `run.json`、逐步 loss、checkpoint/processor；manifest 记录输入与配置身份、参数和完成状态。`--max-steps 1 --epochs 1` 可限制 smoke，不作为收敛或研究效果证据。

Base 和 SFT 使用相同入口、公共 prompt、greedy decode、解析器与评分规则：

```bash
python scripts/evaluate_qwen_episode_sft.py \
  --dataset /path/to/audited-export --dataset-manifest-sha256 SHA256 \
  --model /path/to/Qwen3-VL-2B-Instruct --out output/base-validation \
  --task diagnosis --split validation --max-new-tokens 512 --seed 20261013 \
  --bootstrap-samples 1000
python scripts/evaluate_qwen_episode_sft.py \
  --dataset /path/to/audited-export --dataset-manifest-sha256 SHA256 \
  --model /path/to/Qwen3-VL-2B-Instruct \
  --adapter output/diagnosis-evidence/checkpoint --out output/sft-validation \
  --task diagnosis --split validation --max-new-tokens 512 --seed 20261013 \
  --bootstrap-samples 1000
python scripts/compare_qwen_sft.py \
  --evaluation base=output/base-validation/metrics.json \
  --evaluation evidence=output/sft-validation/metrics.json \
  --out output/comparison.json
```

停止能力另用 `--task stopping` 及独立 checkpoint/结果目录汇总。冻结 Test 只能在对应实验协议许可后显式选择，不作为整理工作的验收数据。

## 评分与结果解释

每个输入保留一条 `predictions.jsonl`，包括原文、解析错误、目标/预测及原物体身份。非法 JSON、非法部件/关节或错误证据引用均保留在总分母中。缺陷 Macro-F1 使用完整 taxonomy 的固定类别集合（没有支持样本的类别 F1 记 0）；同时报告格式有效率、status/exact verdict、部件定位、开放关节定位、证据支持率（合法预测 findings 中引用受支持的比例；无 findings 时 null，与格式有效率一起报告）、Clean 误报与 False pass。已给定唯一关节的案例不计开放定位。Clean 必须有显式审核 `is_clean=true`；缺失时该指标为 unavailable/null，不能将 uncertain 当作 clean。False pass 是对非 pass 审核目标输出 pass，其分母另行报告。

同一物体的多个 corruption/episode 不独立；bootstrap 按原始来源物体重采样并保留其全部案例。少于两个物体时不生成置信区间。每份结果记录数据版本、manifest/输入 SHA、案例数、独立物体数和推理协议；汇总器拒绝 incomplete 或协议不一致结果。训练、评测完成状态以本次实际产物为准，旧 GPU 结果不能代替回归。

真实的小型训练→保存→重载→推理与测试命令/结果见 [smoke 记录](sft-smoke.md)。该流程仅验证基础设施可运行，不证明 Evidence SFT 有效，不交付 Easy→Expert 的正式实验矩阵。

## 本次验收记录

2026-10-25：`.venv/bin/python -m pytest tests/unit` 为 **219 passed**；无 Torch 的基础环境为 **214 passed、5 skipped**（assistant loss/mask 可选测试）。真实 smoke 与实际 compare CLI 均退出 0，`git diff --check` 通过。

验收只覆盖本次基础设施，不包括历史实验的效果结论：

| 验收项 | 本次证据 |
|---|---|
| 无 GPU、无私有数据的单元测试 | 合成 fixtures；基础环境允许明确 skip Torch 可选测试 |
| SHA、路径、来源身份、原物体 split 隔离 | `test_sft_data.py`、`test_oracle_training.py` |
| Evidence 不依据未观察隐藏缺陷或 clean 生成目标 | 曝光标签匹配、伪造 condition、隐藏 clean pass 拒绝测试 |
| Ordinary/Evidence 同公开输入 | 相同 user 消息和图片、不同诊断目标；stopping 目标完全相同 |
| assistant-only mask | 多图、历史、padding、模板 prefix 不一致与空目标测试 |
| Base/SFT 同评测器且不剔除无效预测 | 共享入口与协议；非法 JSON/部件/证据引用、计数与汇总测试 |
| 实际训练→保存→重载→推理 | 独立 CPU smoke，详见运行记录；不代表研究模型质量回归 |
| PR 内容限界 | 逐文件 diff 审查；仅代码、合成 fixture 定义、文本运行摘要和文档 |

代码重构后的 CUDA 2B/4B/8B 正式训练矩阵及 `requirements-qwen.txt` 中
Torch 2.7.1 GPU 环境均未在本轮执行；CPU smoke 的实际依赖版本另见记录。
