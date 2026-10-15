# 正式训练数据集扩充记录

本文持续记录正式 Train/Validation 数据建设；它不替代工程进度文档，也不包含模型训练或冻结 Part 2 Test 评测。本轮产物为 **PartNet-Mobility 35 个独立物体、138 条通过质量门禁的监督样本及对应 Episode**。

## 目标与边界

目标从验证 UniPhys-40K 转换器转为增加可独立划分、类别和关节结构足够多、来源许可与质量证据可追溯的训练物体。正式 Part 2 Test 只用于身份和几何去重，未修改、未渲染选样、未启动评测；本轮也未启动 Qwen3-VL-2B SFT。

本次增量与既有 UniPhys-40K 正式池合计为 **40 个原始物体（Train 31 / Validation 9）**。以下统计除特别注明外只指本次 PartNet-Mobility 增量，避免把旧数据重复算入。

## 数据来源、许可与获取

主来源为 [PartNet-Mobility 官方 Hugging Face 数据集](https://huggingface.co/datasets/sapien-sim/PartNetMobility)，固定 revision `ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f`。数据访问已由上游批准；用途限定为非商业研究/教育，并须同时遵守 [ShapeNet 条款](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)。许可随 catalog 每个资产保存，不能因转换而改写为更宽松许可。

下载器按 `sha256(seed:filename)` 从不超过 5 MiB 的官方 ZIP 中确定性选 160 个，逐包保存 SHA-256、上游 revision、时间和许可链接。未使用第三方镜像。原始下载及解包约 790 MiB。

[GAPartNet 官方仓库](https://github.com/PKU-EPIC/GAPartNet)声明 CC BY-NC 4.0，但完整数据仍需表单申请并接受额外条款；仓库公开的两个示例不足以形成独立扩量批次，且其对象可能源自 PartNet。故本轮只完成来源评估，不混入正式增量。下一阶段须获得完整官方包后按原始身份与 PartNet 联合去重，不能按重新标注后的 ID 当成新物体。

## 转换与质量门禁

[`download_partnet_mobility.py`](../scripts/download_partnet_mobility.py) 负责确定性获取；[`convert_partnet_mobility.py`](../scripts/convert_partnet_mobility.py) 安全解包并将官方 URDF、网格、关节树、轴和限位转换为 `AssetSpec`。转换逐资产执行以下硬门禁：

1. ZIP 哈希与下载清单一致，`meta.json` 的原始 ID/类别完整；仅接受 1–8 个有限 revolute/prismatic 关节。
2. URDF 关节类型与 `mobility_v2.json` 一致；rest 必须为有限范围内的真实零位。
3. rest 几何尺寸与官方 `bounding_box.json` 的三轴排序误差不超过 2%，最大尺度须在 0.02–5 m。
4. 按上游 source identity、ZIP SHA、规范化顶点指纹与冻结 Test 去重；另排除既有正式池中的 PartNet `6635`、`6659`。
5. 用真实浏览器在三视图和运动端点检查依赖部件确实运动，再人工复核方向、静态部件误动和明显穿模。审核结论及 smoke JSON SHA 固化在 [`partnet-mobility-pilot-v1.json`](../configs/partnet-mobility-pilot-v1.json)。

材质文件缺少 `Kd` 时，转换器使用中性顶点色，不再因 `None` 崩溃；这只补可视化缺省值，不改变几何和运动学。对于上下限跨过零且不对称的关节，Episode 协议新增精确 `rest=0` 动作，避免把 `mid` 错当回零。

### 实际筛选结果

160 个下载物体中有 94 个通过结构门禁。66 个结构拒绝原因为：不支持的 joint/mimic 50、活动关节数不在 1–8 内 9、零位不在有效有限行程内 5、缺失 mesh 1、非法数值 `None` 1。随后从 27 个候选类别优先选择混合/多关节的 48 个做浏览器试点，35 个通过、13 个因依赖部件未产生稳定可见且几何一致的运动而拒绝，自动运动门禁通过率为 **72.9%**。该通过率来自优先抽样的候选，不能直接外推整个约 2K 数据集。

最终 35 个物体覆盖 21 类，分布如下：

| 划分 | 物体 | 类别 | revolute | prismatic | 多关节物体 | 混合关节物体 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 28 | 21 | 29 | 11 | 9 | 0 |
| Validation | 7 | 7 | 13 | 7 | 6 | 2 |
| 合计 | 35 | 21（去重） | 42 | 18 | 15 | 2 |

Validation 含 Box、Eyeglasses、Faucet、Refrigerator、StorageFurniture、Table、TrashCan 各一个；这些类别在 Train 均有独立原物体，但同一原物体及其全部变体只属于一个划分。

## 划分与冻结 Test 隔离

`split_group` 是 `PartNet-Mobility:<archive_id>`。每个原始 ZIP、转换程序、clean/单缺陷/组合缺陷样本和全部 Episode 严格继承同一组划分。最终审计确认 35 个 source identity 和 35 个 ZIP SHA 均唯一，Train/Validation 组交集为 0，与冻结 Test 的来源身份及几何指纹交集为 0；冻结 Test 清单 SHA-256 为 `3623a753cdcf5063090e91e414f335b08cce9d38e57ca63e1a7eb04e0a44eb99`。

完整资产产物位于 `data/assets/partnet-mobility-trainval-v1/`：`catalog.json`、`splits.json`、35 份 spec 和含 160 个对象逐项结论的 `conversion.json`。这些大文件按仓库约定不进 Git，但均保留在当前工作区。

## 监督样本与 Episode

冻结配置 [`generate_partnet_mobility_supervision.yaml`](../configs/generate_partnet_mobility_supervision.yaml) 生成 140 条样本，覆盖 clean、`structure.part_offset`、`kinematics.joint_axis` 以及二者组合。注入记录只是候选标签；质量门禁重新计算目标约束是否真实被破坏，并逐缺陷做隔离检查，不因“已请求注入”就自动接纳。该检查仍是合成数据的程序化 Gold，不冒充自然错误的独立人工 Gold。

| 项目 | 结果 |
| --- | ---: |
| 生成/编译成功 | 140 / 140 |
| 通过 / quarantine | 138 / 2 |
| Train / Validation（通过项） | 110 / 28 |
| clean / 单缺陷 / 组合缺陷（通过项） | 47 / 42 / 49 |
| part offset / joint axis 标签（通过项） | 72 / 68 |

两条 `PM_100054` joint-axis 样本因扰动未真实违反目标约束而隔离，没有进入 gold 或 Episode。prepared 私有 gold 的 138 行全部保留 `formal_training=true`、source identity、archive/spec SHA 与许可。

`oracle-first-failure-v3/` 是第一轮 checkpoint 使用的历史导出；`oracle-first-failure-v4/` 只重算 assistant 与逐前缀停止目标，复用同一 138 个样本、429/110 行 Train/Validation Episode、私有审计和全部图像。观察、Gold 与像素均未重生成。特权 counterfactual 教师仍只用于既有动作轨迹和私有审计，动作标签不进入 SFT，**不是模型能力、人工 Gold 或可观测最优策略证明**。

自然生成错误单独登记在 [`partnet-mobility-natural-errors-v1.json`](../configs/partnet-mobility-natural-errors-v1.json)。本轮没有启动付费生成调用，故数量明确为 0、状态为 `not_collected`；没有用 corruption 注入标签冒充自然错误或独立人工 Gold。后续只有在人工不读取注入标签、依据真实渲染独立标注后，才可进入该集合。

机器可读总审计位于 `data/benchmarks/partnet-mobility-formal-v1/dataset-audit.json`，由 [`audit_training_dataset_expansion.py`](../scripts/audit_training_dataset_expansion.py) 从 catalog、split、raw/prepared manifest 与 Episode 重新计算，不依赖本文手填统计。

## SFT 数据就绪性验证

### 数据版本与输入协议

本节记录第一轮训练前对 `oracle-first-failure-v3` 的管线验证；其 manifest SHA-256 为 `f95302809fa135d2d161577dee9b44ff405a671287ae07bc1015f40a8e1f546b`。后续语义审计发现 v3 的 assistant 解释仍引用私有状态，因此“输入未含私有字段”并不等于“目标文本无泄漏”，v3 不再是新训练的数据版本。Qwen 配置 SHA-256 为 `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b`；正式资产池仍为 PartNet 35 个加 UniPhys 5 个，共 Train 31 / Validation 9。

SFT 输入严格为「参考图在前、截至当前停止点的历史观察图随后、公开 verifier 协议与 JSON payload」；payload 仅允许 `context/remaining/history/image_order/taxonomy_ids/training_objective`。审计逐行把公开 history 与私有轨迹的同长度前缀作精确比较，因此未来观察不可能进入输入；私有 corruption、exposure、候选程序、参考程序和 Oracle 信号均不在消息中。私有 annotation 只在离线审计侧核对 source、candidate/reference SHA 及可见证据，不传给 processor。训练 loader 只选择 terminal verdict，401 条特权 Oracle 动作目标全部排除；clean 隐藏真值也不生成 `pass`。

### 全量审计与分布

[`audit_sft_readiness.py`](../scripts/audit_sft_readiness.py) 实际读取 539 行 Episode 及其 1,673 次图像引用，校验 539 个实际 PNG 路径、264 个不同内容 SHA；138/138 个 terminal Episode 均通过 `formal_training`、上游 source identity/content SHA、candidate/reference SHA、原始资产/来源组划分、objective、因果历史、图像顺序和结构化 verdict 校验，失败案例为 0。联合 40 资产的 Train/Validation split group、source identity 均无交叉；与冻结 Test 的 source identity 无交叉，PartNet 还核对转换时使用的 Test 清单 SHA 及逐资产 source/几何去重阴性记录。未读取 Test 图像、未渲染 Test、未做模型评测，也未修改冻结清单。

| 分布 | Train | Validation | 合计 |
| --- | ---: | ---: | ---: |
| terminal Episode | 110 | 28 | 138 |
| `fail` / `uncertain` / `pass` | 72 / 38 / 0 | 19 / 9 / 0 | 91 / 47 / 0 |
| 目标中的 part offset / joint axis finding | 57 / 17 | 15 / 4 | 72 / 21 |
| 原始注入 part offset / joint axis 标签 | 57 / 54 | 15 / 14 | 72 / 68 |
| `evidence_sufficient` / `budget_exhausted` | 72 / 38 | 19 / 9 | 91 / 47 |

clean / 单缺陷 / 组合缺陷为 47 / 42 / 49；Episode 长度 1 / 2 / 6 分别为 63 / 28 / 47，对应输入图像数 2 / 3 / 7。全部目标的 `training_objective` 都是 `first_failure`。这也揭示出明确偏置：47 个 clean 例全部保守监督为 `uncertain`，没有 `pass`；first-failure 目标覆盖全部 72 个 offset 标签，却只保留 68 个 axis 标签中的 21 个可见首失败 finding；缺陷仍只有两类、自然错误人工 Gold 为 0，且 Validation 只有 7 个 PartNet 原始物体。

真实 Qwen processor 对 Train 110 / Validation 28 全部成功，序列长度 1,438–2,574 token，assistant 监督 67–250 token；所有 prompt/image token 均为 `-100`，assistant 标签连续完整，无截断或多图错位。训练入口本身逐样本构造 batch（batch size 1、无 padding collator），本次复用完全相同的 `tokenize_example`，各取一条 7 图 Train/Validation 样本执行 `eval + inference_mode` 前向：loss 分别为 `2.3936724663` 与 `2.4154891968`，均有限。该数值只证明 batch 能进入模型和 loss，不作为训练或质量指标。

机器可复核报告 `output/partnet-mobility-sft-readiness-v1.json` 只证明 v3 可被 processor/loader 消费；其 SHA-256 为 `256c83ad164a74034a6fde0d8eccd7c220a02dcb5e34aaf6b046f523490d280a`。它不再构成监督语义就绪证明。修复后的 v4 语义审计见下文；仍不支持动作 SFT、`pass` 校准或泛化结论。

复现命令（只读数据与模型；`--forward` 不训练）：

```bash
python -m scripts.audit_training_dataset_expansion \
  --out data/benchmarks/partnet-mobility-formal-v1/dataset-audit.json

CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python -m scripts.audit_sft_readiness \
  --benchmark data/benchmarks/partnet-mobility-formal-v1 \
  --oracle data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 \
  --catalog data/assets/partnet-mobility-trainval-v1/catalog.json \
  --asset-pool data/assets/partnet-mobility-trainval-v1 \
  --asset-pool ../TurnItOver/data/assets/uniphys40k-trainval-v3 \
  --test-plan configs/uniphys-test.json \
  --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --forward \
  --out output/partnet-mobility-sft-readiness-v1.json
```

## 第一轮 Qwen3-VL-2B 正式 SFT 受控对照

### 设计与监督差异

三组均使用同一 Qwen3-VL-2B-Instruct、同一 110 Train / 28 Validation terminal Episode、相同公开输入图像与历史，以及 Validation `max_new_tokens=512` 的 greedy decode。两组 LoRA 都用 seed `20261013`、3 epochs、330 optimizer steps、AdamW `2e-4`、rank 8、alpha 16、dropout 0，并只训练 `q/k/v/o_proj`。训练配置保存在 [`partnet-mobility-sft-v1.json`](../configs/partnet-mobility-sft-v1.json)。底座 config SHA 与上述就绪审计一致，Train/Validation JSONL SHA 分别为 `a04446c7…80ca1` / `0e9eeeab…9405`。

两组 110 条样本的用户消息、图像、历史和 terminal 前缀逐条完全相同，差异只在 assistant verdict。普通组用独立 corruption Gold 构造完整诊断，共 111 个 finding；证据组只保留当前公开观察下通过既有 exposure 门禁的 74 个 finding。旧审计逐 ID 列出的37条普通组独有 finding 全为 `kinematics.joint_axis`。但最新逐字段复核还发现 v3 的 `summary/limitations` 会随私有 Gold 改变：无可见 finding 时仍写入 injected/private-label 等信息。因此以下两个 checkpoint 都是**旧监督**的分析对象，尤其证据 checkpoint 不能代表修复后目标的训练效果。

两组训练均已完成，`run.json` 为 `complete`，`loss.jsonl` 各有连续330步，checkpoint 完整：

| 训练 | finding | epoch mean loss 0 / 1 / 2 | adapter SHA-256 |
| --- | ---: | --- | --- |
| 普通 SFT | 111 | 0.391 / 0.181 / 0.180 | `dd473181652e370dbe7d6e5d7a7e0852bfe0e30bb6cb0be4dfe69b77b51b227c` |
| 证据约束 SFT | 74 | 0.382 / 0.166 / 0.161 | `58ed2f0f3b246ffae5781104be11209333b5af232717315b3ccf10847b1be37f` |

训练 loss 只用于产物完整性检查，不作为监督更优的证据。checkpoint、配置、逐步曲线分别位于 `output/partnet-qwen3vl2b-{ordinary,evidence}-sft-v1/`，按仓库规则不提交大型权重。

### 三组 Validation 实际结果

三组各28条逐案例推理全部完成且合法，重跑结果与此前预测逐字段一致。macro-F1 只在有 Gold 支持的 `fail/uncertain` 两类上计算，不能扩展成三分类或 `pass` 校准结论。定位 finding 以 `(defect_id, joint_id, parts)` 精确匹配。

| 指标 | 底座 | 普通 SFT | 证据约束 SFT |
| --- | ---: | ---: | ---: |
| 合法输出 / terminal verdict | 28/28 | 28/28 | 28/28 |
| status accuracy | 25.0% | 100% | 100% |
| status macro-F1 (`fail/uncertain`) | 0.200 | 1.000 | 1.000 |
| Gold `fail` → `fail/uncertain/pass` | 0 / 19 / 0 | 19 / 0 / 0 | 19 / 0 / 0 |
| Gold `uncertain` → `fail/uncertain/pass` | 0 / 7 / 2 | 0 / 9 / 0 | 0 / 9 / 0 |
| fail false-positive rate | 0% | 0% | 0% |
| exact verdict + 定位 | 7/28 | 9/28 | 13/28 |
| finding precision / recall / F1 | n/a / 0 / 0 | 0.263 / 0.526 / 0.351 | 0.375 / 0.474 / 0.419 |
| defect-type F1 | 0 | 0.667 | 0.837 |
| exact part/joint localization F1 | 0 | 0.351 | 0.419 |
| unsupported / predicted findings | 0 / 0 | 33 / 38 | 15 / 24 |

status 的满分存在明确捷径：19个 `fail` Validation 例都在第1/2步形成 terminal，9个 `uncertain` 都在第6步预算耗尽，而 `history/remaining` 是公开输入。因此两组的 100% status/macro-F1 只能说明学会当前 terminal 数据分布，**不能说明学会视觉提前停止**。这里的28/28只称 `valid_output_rate`；真实停止时机由下文逐前缀指标单独评测，不再使用 `stop_accuracy` 指代合法输出率。

证据约束的实际收益集中在抑制过度诊断：unsupported finding 减少18个（33→15），exact 提高4例，finding precision/F1 与 defect-type F1 上升。普通组对19个 `fail` 全都预测 axis，实际 axis Gold 仅4例；证据组把 axis 预测从19例降到7例，axis FP 从15降到3。退化也存在：总体 TP 从10降到9、recall 从0.526降到0.474；offset 精确定位 TP 从6降到5、F1 从0.353降到0.313。证据组仍在17例中预测 offset（Gold 15例），所以此前的 `fail/offset` 模板化倾向没有消失。

按7个独立 Validation 原始物体汇总如下，单元格为 `status正确 / exact / unsupported findings`：

| 原始物体 | 底座 | 普通 SFT | 证据约束 SFT |
| --- | --- | --- | --- |
| PM_100202 | 2 / 2 / 0 | 4 / 2 / 4 | 4 / 2 / 2 |
| PM_102252 | 0 / 0 / 0 | 4 / 0 / 7 | 4 / 2 / 2 |
| PM_102590 | 1 / 1 / 0 | 4 / 1 / 4 | 4 / 1 / 3 |
| PM_10797 | 1 / 1 / 0 | 4 / 3 / 1 | 4 / 3 / 1 |
| PM_1280 | 0 / 0 / 0 | 4 / 0 / 7 | 4 / 1 / 3 |
| PM_34610 | 0 / 0 / 0 | 4 / 0 / 8 | 4 / 1 / 3 |
| PM_45759 | 3 / 3 / 0 | 4 / 3 / 2 | 4 / 3 / 1 |

正式产物为 `base-val-v5`、`ordinary-val-v4`、`evidence-val-v3` 的 `predictions.jsonl/metrics.json` 及 `output/partnet-qwen3vl2b-controlled-comparison-v5.json`；对照报告 SHA-256 为 `92e8abc8f8d420c0e30bf9fb7107be45c5af30e09d4d9922da16ff6d30f062e5`。两次新增指标初跑在完成推理后因汇总代码错误标记为 incomplete，目录 `base-val-v4` / `ordinary-val-v3` 明确不属于正式结果。

### 实际命令、局限与下一步

```bash
python -m scripts.audit_sft_supervision \
  --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 \
  --out output/partnet-qwen3vl2b-supervision-audit-v1.json

CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/train_qwen_episode_sft.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --out output/partnet-qwen3vl2b-ordinary-sft-v1 --supervision ordinary --seed 20261013 --epochs 3 --learning-rate 2e-4 --lora-rank 8 --lora-alpha 16
CUDA_VISIBLE_DEVICES=1 ../TurnItOver/.venv-qwen/bin/python scripts/train_qwen_episode_sft.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --out output/partnet-qwen3vl2b-evidence-sft-v1 --supervision evidence --seed 20261013 --epochs 3 --learning-rate 2e-4 --lora-rank 8 --lora-alpha 16

CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/evaluate_qwen_episode_sft.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --out output/partnet-qwen3vl2b-base-val-v5 --max-new-tokens 512
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/evaluate_qwen_episode_sft.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --adapter output/partnet-qwen3vl2b-ordinary-sft-v1/checkpoint --out output/partnet-qwen3vl2b-ordinary-val-v4 --max-new-tokens 512
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/evaluate_qwen_episode_sft.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --adapter output/partnet-qwen3vl2b-evidence-sft-v1/checkpoint --out output/partnet-qwen3vl2b-evidence-val-v3 --max-new-tokens 512
python scripts/compare_qwen_sft.py --base output/partnet-qwen3vl2b-base-val-v5/metrics.json --ordinary output/partnet-qwen3vl2b-ordinary-val-v4/metrics.json --evidence output/partnet-qwen3vl2b-evidence-val-v3/metrics.json --out output/partnet-qwen3vl2b-controlled-comparison-v5.json
```

三组对照已真正完成，但只有单 seed、7个 Validation 原物体、两类合成缺陷和0个可信 `pass`；既不能估计方差或跨来源泛化，也不能声称 pass 校准。下一轮先做多 seed、按 assistant token 匹配预算和单/组合缺陷分层，并设计不让 terminal 长度直接泄露 status 的停止消融；方案冻结前继续不使用 Part 2 Test。

## 第一轮 SFT 收益与预算捷径诊断

### 评测设计与可识别边界

2026-10-14 复用同一底座、普通 LoRA、证据约束 LoRA 和原28条 Validation Episode，未训练或修改任何 checkpoint。诊断脚本对导出的 **110条真实因果前缀**逐条推理；输入只含当时已经发生的历史及对应图像，不拼接未来观察。私有 label、exposure、witness index 与 Episode `evidence` 只在输出后评分，不进入 Qwen processor。另对28条原 terminal 前缀只删除公开 JSON 中的 `remaining` 键，历史、图像、顺序和其余文本保持不变；这是分布外敏感性测试，不是伪造预算值。

这28条 Episode 实际只覆盖7个 PartNet Validation 原始物体，而不是正式资产池的9个 Validation 物体；另2个 UniPhys Validation 物体没有进入本批 PartNet Episode 或这三个 checkpoint 的评测。因此本节只按7个实际原物体汇总，不能补写成9物体结论。数据中还满足 `remaining = 6 - history_length` 且 `image_count = history_length + 1`，三种阶段线索完全共线。删除 `remaining` 后仍留下历史长度和图像数，所以各线索的独立因果贡献不可识别。

### 同预算真实前缀与预算字段消融

真实数据提供了两组合法的同预算比较：decision 1 / remaining 5 有证据10条、无充分证据18条；decision 2 / remaining 4 两边各9条。三组模型的 status/动作结果如下，括号为 unsupported finding 数：

| 阶段 | 可见证据 | 底座 | 普通 SFT | 证据约束 SFT |
| --- | --- | --- | --- | --- |
| step 1 / remaining 5 | 充分，10条 | 10 uncertain（0） | 10 fail（15） | 10 fail（7） |
| step 1 / remaining 5 | 不充分，18条 | 18 uncertain（0） | 18 fail（36） | 18 fail（18） |
| step 2 / remaining 4 | 充分，9条 | 9 uncertain（0） | 9 fail（18） | 9 fail（8） |
| step 2 / remaining 4 | 不充分，9条 | 9 uncertain（0） | 9 fail（18） | 9 fail（17） |

两组 SFT 在相同预算阶段完全不按证据是否充分改变 status，证据组只是少输出了一部分 finding。更早的 step 0 上，底座给出23个合法动作、5个非法决定；两组 SFT 均为28/28非法 verdict，因为无观察却试图引用 finding。对9条一直无充分证据的长前缀，普通组在 step 3/4/5 分别产生2/5/6个 `fail`，其余因错误引用 evidence step 非法；证据组在 step 3–5 均为9/9 `fail`；两组只有到 step 6 / remaining 0 才全部变成 `uncertain`。因此 terminal-only 的“合法停止率”确实掩盖了前缀上的模板化提前 fail，不能解释为学会视觉停止。

只删除 `remaining` 后，底座26/28保持相同输出，2个 `pass` 转为 `uncertain`；普通组只有4/28文本完全相同，5/9个原 `uncertain` 转为 `fail`，status正确从28降至23、unsupported 从33增至43；证据组只有3/28文本完全相同，9/9个原 `uncertain` 全转为 `fail`，status正确从28降至19、unsupported 从15增至33。这直接支持“`remaining` 字段显著参与两组 SFT 的 terminal status 决策”，但不能证明模型只读该字段，也不能区分历史长度、图像数量及历史内容的贡献。

### 定位收益、代价与集中度

原 terminal 输入上的结果与上一节一致。按当前可见 Gold finding 精确匹配，axis 与 offset 的分解为：

| 模型 / finding | TP | FP | FN | Precision | Recall | F1 | 预测案例数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通 / joint_axis | 4 | 15 | 0 | .211 | 1.000 | .348 | 19 |
| 证据 / joint_axis | 4 | 3 | 0 | .571 | 1.000 | .727 | 7 |
| 普通 / part_offset | 6 | 13 | 9 | .316 | .400 | .353 | 19 |
| 证据 / part_offset | 5 | 12 | 10 | .294 | .333 | .313 | 17 |

axis 改善来自保留4个 TP 同时去掉12个 FP，支持“证据约束抑制普通监督引入的 axis 过度诊断”；offset 则 TP 6→5、FN 9→10、F1 .353→.313，确认漏检代价。证据组仍在17例预测 offset，而且在同预算无充分证据前缀仍持续输出 fail/offset，因此 offset 模板偏置没有消失。预测总量下降本身不算能力提升；只有与 TP/FP/FN、unsupported 和配对 exact 一起解释。

按 corruption 组合，普通→证据组分别为：clean 的 exact 均为9/9；axis-only 的 exact 0→1、TP/FP/FN 4/4/0→4/3/0、unsupported 8→3；axis+offset 的 exact 0→3、3/17/7→3/7/7、unsupported 17→7；offset-only 的 exact 仍为0、3/7/2→2/5/3、unsupported 8→5。收益主要来自压低组合缺陷里的额外 finding，不是 recall 增长。

按原物体配对的“证据组减普通组”如下：

| 原物体 | exact 差 | unsupported finding 差 |
| --- | ---: | ---: |
| PM_100202 | 0 | -2 |
| PM_102252 | +2 | -5 |
| PM_102590 | 0 | -1 |
| PM_10797 | 0 | 0 |
| PM_1280 | +1 | -4 |
| PM_34610 | +1 | -5 |
| PM_45759 | 0 | -1 |

4个 exact 增益集中在3/7个物体，但 unsupported 降低覆盖6/7个物体，并非单一物体造成。以原物体为 cluster、seed `20261013` 做10,000次描述性 bootstrap，exact accuracy 配对差为 +.143，95% percentile 区间 `[+.036, +.286]`；每案例 unsupported 差为 -.643，区间 `[-1.000, -.286]`。只有7个 cluster、单一训练 seed，区间不稳定，不能替代多 seed 或更大独立 Validation。

### 监督语义修复与停止目标

v4 对539行 Episode 逐项重审 `verdict/reason/findings/evidence_steps`，只改 assistant 目标：401条动作不变，138条 terminal 解释变化；status、findings、evidence steps 均0变化，summary变化47条、limitations变化138条。修复后 assistant 中0条命中 `injected/private label/hidden defect/unexposed defect`，无证据 terminal 只剩1种公开事实解释，539个精确公开前缀组0冲突。私有 Gold、完整性和注入记录仍保留在 `annotations.private.jsonl`，不进入模型消息。审计报告为 `output/partnet-qwen3vl2b-supervision-audit-v2.json`（SHA-256 `2ccbdc354f5d7ef7a8625a124ab666fed4e9d3f2aef92bd96b2eab024e9cd176`）。

另导出539条逐前缀停止监督：401 `continue`、91个有证据 `stop/fail`、47个预算或观察空间耗尽的 `stop/uncertain`。`first_failure` 在首个合格 finding 即可停止；`comprehensive_within_budget` 必须继续到该策略的最终前缀，二者不混用。当前 v4 实跑与下述评测只覆盖 `first_failure`；`comprehensive_within_budget` 尚无对应模型结果。预算特征并不能完美决定新 stop 标签：step 1/remaining 5 和 step 2/remaining 4 都同时含 `continue` 与 `stop`。

### 无图像预算基线与停止时机

无图像基线只使用 `step/remaining/history_length/image_count`，在 Train 拟合确定性多数查找并评测相同110条 Validation 前缀；脚本不读取图像字节。它的 terminal status 为28/28，已经完全复现两组 SFT 的 status 满分；terminal defect-type exact 为23/28、F1=.737，也说明缺陷类型结果很大一部分可能来自阶段分布。它不具备部件定位能力。逐案例位于 `output/partnet-no-image-budget-baseline-v1/predictions.jsonl`（SHA-256 `86650166…88a`）。

| 条件 | terminal status | finding F1 | defect-type F1 | exact verdict | 首次停止 exact / early / missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| 底座 | 7/28 | 0 | 0 | 7/28 | 10 / 18 / 0 |
| 普通 SFT（旧 v3） | 28/28 | .351 | .667 | 9/28 | 10 / 18 / 0 |
| 证据 SFT（旧 v3） | 28/28 | .419 | .837 | 13/28 | 10 / 18 / 0 |
| 无图像预算基线 | 28/28 | 0 | .737 | 9/28 | 9 / 0 / 19 |

这里将 `valid_output_rate`、诊断正确率和停止时机严格分开。两组 SFT 的逐前缀 stop-decision accuracy 均为28/110，且都在18/28个 Episode 提前停止；它们没有学会正确停止。预算基线 stop-decision accuracy 为91/110，但漏停19/28个应停止 Episode，不能把高逐前缀准确率解释为正确时机。

### 结论、产物与复现

仍成立的描述性收益是：在同一旧 checkpoint 和同一28例 terminal 输入上，证据 SFT 比普通 SFT 少18个 unsupported finding、exact 多4例，并改善 axis 与精确部件定位汇总。不能继续主张的部分是：status 收益已被无图像基线完全解释，defect-type 也高度受阶段捷径影响；同阶段前缀上两组 SFT 对证据充分/不足均输出 `fail`，且停止时机失败，因此现有结果不证明依赖视觉证据。旧 checkpoint 训练于 v3 泄漏目标，**不是修复后监督的训练结果**。

机器汇总 `output/partnet-qwen3vl2b-diagnostic-analysis-v2.json`（SHA-256 `3b8fa8099facbc31b652ae089be3e70e5cab84f15ddc0cc071f33f4ec50a3cf7`）含逐物体、缺陷组合、同阶段对照、stop confusion 和全部输入 SHA。无法比较的条件明确为：0个可信 pass、0个自然错误人工 Gold、2个正式 Validation 物体无 Episode、未评测 comprehensive 策略、预算基线不能定位部件。冻结 Part 2 Test 未读取，Gold/证据标准未修改。

```bash
python -m scripts.reexport_episode_targets --source data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --out data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4
python -m scripts.audit_sft_supervision --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4 --baseline-dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --out output/partnet-qwen3vl2b-supervision-audit-v2.json
python -m scripts.evaluate_budget_baseline --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4 --out output/partnet-no-image-budget-baseline-v1
python -m scripts.analyze_qwen_episode_diagnostics \
  --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4 \
  --budget-baseline output/partnet-no-image-budget-baseline-v1/predictions.jsonl \
  --base-prefix output/partnet-qwen3vl2b-diagnostic-base-prefix-v1/predictions.jsonl \
  --ordinary-prefix output/partnet-qwen3vl2b-diagnostic-ordinary-prefix-v1/predictions.jsonl \
  --evidence-prefix output/partnet-qwen3vl2b-diagnostic-evidence-prefix-v1/predictions.jsonl \
  --base-no-remaining output/partnet-qwen3vl2b-diagnostic-base-no-remaining-v1/predictions.jsonl \
  --ordinary-no-remaining output/partnet-qwen3vl2b-diagnostic-ordinary-no-remaining-v1/predictions.jsonl \
  --evidence-no-remaining output/partnet-qwen3vl2b-diagnostic-evidence-no-remaining-v1/predictions.jsonl \
  --out output/partnet-qwen3vl2b-diagnostic-analysis-v2.json --seed 20261013 --bootstrap-samples 10000
```

下一次训练需要阶段线索匹配且跨物体平衡的证据充分/不足前缀、真实 `continue/stop` 样本、分别冻结的 `first_failure` 与 `comprehensive_within_budget` 策略，并补充 pass、自然错误、更多独立物体和多 seed。训练前应先证明同一 step、remaining、历史长度和图像数下目标仍可由可见内容区分；本轮不重新训练。

## API 交互教师最小闭环

[`run_interactive_teacher.py`](../scripts/run_interactive_teacher.py) 统一 Mock、本地 Qwen3-VL-2B 和 Kimi 的模型可见请求：仅包含参考图、已发生的观察历史、合法动作空间、`remaining` 和与其一致的 `stop_required`。动作在环境执行后才回填；每次原始输出、request JSON、图像顺序、token、预算和异常均保存，Episode ledger 原子落盘。动作已选但环境中断时保存 `pending_action`，恢复会先重放既有状态再执行该动作，不重复调用教师。独立 `audit.json` 分开报告 Pipeline 完整性、是否收到教师响应、教师动作/输出合法性、verdict 是否存在和诊断证据支持；私有 Gold 仅在结束后比较。预算归零仍未得到合法 verdict 时安全结束为 `budget_exhausted=true, teacher_verdict_missing=true`，Episode 不写入伪造 verdict。

2026-10-14 在 Train Episode `f32240d5152e8b45fd0fcb02` 上停止于最小验证规模，全部图像为256×256。Mock 真实 Chromium 闭环执行 `joint_0→limit`，随后在 `remaining=0, stop_required=true` 返回合法 `uncertain` verdict，`pipeline_ok=true`、预算1/1；观察与参考 SHA 不同，但 verdict 没有诊断 finding。本地 Qwen 在预算2内先选 `front`，看到更新历史后再选 `joint_0→limit`，两次均合法执行且取得不同图像；预算归零后连续三次仍请求 `joint_state`，因此安全收口为 `budget_exhausted`、明确缺失 verdict，未伪造诊断。Pipeline 与环境链路正常，但该模型的停止输出不合法、无诊断证据结论；共5次本地推理，4,662 input + 78 output token，费用0元。单测另覆盖教师提前合法停止、零预算合法 verdict、零预算缺失 verdict、非法/不可解析输出、执行异常，以及中断恢复不重复选动作。

PartNet-Mobility 官方条款允许获批研究者将数据库用于非商业研究与教育，并要求获得数据库访问的研究同事先同意相同条款；但没有明确禁止把单张渲染图提交给第三方模型做非商业研究推理（虽然这不等同于上游对云处理的专项书面许可）。按本轮采用的“无明确禁止即可做最小测试”边界，2026-10-14 从正式 Train split 选取 `f32240d5152e8b45fd0fcb02`（`PM_100054`，`formal_training=true`），未使用 Validation 或冻结 Test。Kimi `k3-256k` 成功接收一张 PartNet-Mobility 参考图、空观察历史、`remaining=2` 和 `stop_required=false`，以 `finish_reason=stop` 返回合法 `request_view(front)`；系统正确解析并执行动作，生成与参考图哈希不同的新观察，随后按单次请求上限停止，未重试或切换模型。该正式 Train 案例的单步鉴权、多模态、协议解析、环境执行和审计链路结论为 `READY`；非空观察历史的二次多图回传尚未实测。本次用量599 input、513 output tokens（其中487 reasoning），网关账单费用未知。

此前项目自建 `toy_cabinet` compatibility fixture 用于定位兼容性问题：前四次用户授权请求纠正端点并定位 HTTP 401/403；后续 `/models` 探针将网关403归因为 Cloudflare Error 1010 对 Python `urllib` 默认 browser signature 的屏蔽。仅为 `openai_compatible` 增加稳定 User-Agent 后请求到达模型，放开输出上限后得到合法动作。累计真实请求7次。机器汇总为 [`interactive-teacher-v1-results.json`](../output/interactive-teacher-v1-results.json)，可提交的脱敏证据为 [`kimi-partnet-real-compat-v1.json`](evidence/kimi-partnet-real-compat-v1.json)；含图片的完整本地证据留在被忽略的 `output/interactive-teacher/`，不进入提交。

```bash
python -m scripts.run_interactive_teacher --mode mock --budget 1 --out output/interactive-teacher/mock-v2
CUDA_VISIBLE_DEVICES=1 ../TurnItOver/.venv-qwen/bin/python \
  -m scripts.run_interactive_teacher --mode local-qwen --budget 2 --max-tokens 384 \
  --playwright-site-packages /home/laq/anaconda3/lib/python3.12/site-packages \
  --out output/interactive-teacher/local-qwen-v3
python -m scripts.run_interactive_teacher --mode kimi \
  --sample-id f32240d5152e8b45fd0fcb02 --budget 2 \
  --third-party-transmission-reviewed \
  --transmission-review-basis partnet-noncommercial-terms-no-explicit-render-api-prohibition \
  --estimated-input-tokens 4096 \
  --out output/interactive-teacher/kimi-partnet-real-compat-v2
```

第二条命令只把已有 Playwright 安装追加到缺少该包的 Qwen 环境；Qwen 权重仍由原本环境离线加载。Kimi 模型和凭证由 `.env` 的独立 Teacher 配置提供；外发审阅依据显式进入 Episode。网关计费未知时不填人民币单价，费用记录为 `null`，不能写成0元。适配器默认最多一次，并可为有界 Episode 显式提高上限；始终不重试、不切换模型。本轮没有 Generator API 采集、训练、冻结 Test 访问或规模扩张。

### 第一轮 Kimi 交互教师采集（2026-10-14）

按 Issue #4 的停止条件，从不同 PartNet-Mobility Train 原物体选择 clean `af10f7137bbe158aa03557b5`（`PM_100934`）与可控缺陷 `f32240d5152e8b45fd0fcb02`（`PM_100054`，part offset + joint axis）。两者均为 `formal_training=true`；本轮沿用已审阅的非商业研究外发边界，并将审阅依据写入 Episode。每条预算1、最多2次 Kimi 请求、非法输出不重试，两个 Episode 串行执行；累计第4次真实请求后停止。

结果为 **0/2 完整 Episode、2/2 合法动作、1/2 合格动作、0 个有证据 finding、0 个 structured unsupported finding、2 个无效终判、2 个 unsupported terminal claim**。clean 条选择 `front`，所得图像与参考逐字节相同，没有新增信息；缺陷条选择 `oblique_fr` 并产生新图像，故保留为一条候选动作监督，但没有形成合格诊断。两次最终请求都返回了正常 HTTP/model response，却没有遵循 `Verdict(status, confidence, findings, summary, limitations)`：clean 使用顶层字符串 `verdict="pass"`，缺陷条使用 `label/corrupted/rationale`，均被既有 parser 拒绝并真实记为 `teacher_verdict_missing=true`。缺陷条还尝试判定 accept，与独立 Gold 相反；clean 的 pass 也不能由一次重复观察证明。两条因预算耗尽而停止是执行层合理收口，不代表终判合理。

实际 token 为 clean 1,475 input + 1,928 output、缺陷 1,466 input + 3,524 output，合计 **8,393 token / 4次调用**；网关账单未知，费用保持 `null`。另用同一 Teacher runner 的 candidate auto-framing、相同资产和预算1离线执行仓库现有 fixed/random acquisition，模型调用为0：fixed 首步为 `runtime:stats`，random seed 0 为 `oblique_fr`。clean 上 Kimi 的重复 front 不如 random 的新视角；缺陷上 Kimi 与 random 的动作和图像 SHA 完全相同，未取得取证优势。离线对照不请求 verdict，因此不伪造诊断比较。既有特权 Oracle export 使用 sample framing，与本轮 Teacher 的 candidate auto-framing 不同，其 exposure lookup 只作背景信息，不作为严格像素匹配对照。两条终判均不可训练，因此本轮相对现有特权 Oracle **没有形成足以扩大采集的新增监督**。结论是暂不扩大付费采集；下一阶段先在离线测试中把完整 verdict JSON schema 显式放入教师 prompt 并验证结构化输出合同，再重复同规模门禁，而不是放宽 parser 或证据规则。

完整 prompt、可见 request、图片顺序、原始 response、状态变化、token 和审计保存在本地忽略目录 `output/interactive-teacher/kimi-pilot-round1-{clean-pm100934,defect-pm100054}/`；可提交的逐文件 SHA、原始 JSON 输出、离线对照和汇总见 [`kimi-interactive-teacher-round1.json`](evidence/kimi-interactive-teacher-round1.json)。没有调用 Generator、训练 SFT、访问冻结 Test 或修改证据规则。

独立审计分两层：确定性仓库审计与同 runner 离线回放已完成；额外启动的 fresh same-family 审计代理在确认“两条终判均不合法”“prompt 未展开 verdict schema”“Oracle/Teacher framing 不同”后触发用量上限，未完成 A–H 全项，因此机器记录为 `acceptance_status=blocked`，不冒充独立 PASS。

```bash
python -m scripts.run_interactive_teacher --mode kimi \
  --sample-id af10f7137bbe158aa03557b5 --budget 1 --max-real-requests 2 \
  --estimated-input-tokens 8192 --third-party-transmission-reviewed \
  --transmission-review-basis partnet-approved-noncommercial-research-terms-no-explicit-render-api-prohibition \
  --out output/interactive-teacher/kimi-pilot-round1-clean-pm100934
python -m scripts.run_interactive_teacher --mode kimi \
  --sample-id f32240d5152e8b45fd0fcb02 --budget 1 --max-real-requests 2 \
  --estimated-input-tokens 8192 --third-party-transmission-reviewed \
  --transmission-review-basis partnet-approved-noncommercial-research-terms-no-explicit-render-api-prohibition \
  --out output/interactive-teacher/kimi-pilot-round1-defect-pm100054
python -m scripts.compare_teacher_acquisition \
  --sample-id af10f7137bbe158aa03557b5 \
  --sample-id f32240d5152e8b45fd0fcb02 --seed 0 \
  --out output/interactive-teacher/kimi-pilot-round1-baselines
```

### 第二轮 Kimi 闭环、独立审核与导出（2026-10-15）

本轮保留 `4997c59` 和第一轮原始目录不变，修正了三个闭环问题：教师 prompt 现在展开完整 action/verdict/finding schema 与 `fail/pass/uncertain` 边界；非法输出通过既有 parser 产生可见反馈并在 `--decision-attempts` 内有限重试，`--max-real-requests` 同时约束重试；真实 runner 从 raw Train Sample 恢复与 Oracle 相同的冻结 framing，不再用候选自动 framing。请求仍只含 context、参考图、已执行历史、剩余预算、合法 taxonomy 和错误反馈，不含 Gold、候选程序、counterfactual exposure 或未来观察。

独立审核读取正式 Train 的身份 SHA、prepared checker-qualified Gold 和 `oracle-first-failure-v4` 的 leave-one-corruption-out exposure。`fail` 只有在 finding 与客观缺陷一致、且每个 `evidence_steps` 都映射到暴露该缺陷的冻结 witness 时才通过；广义 scope 的 `pass` 不会因 clean 私有标签而通过；`uncertain` 若漏掉当前 witness 已暴露的客观缺陷则拒绝。审核逻辑和训练导出分别位于 `turnitover/teacher/evidence.py` 与 `turnitover/teacher/export.py`，避免把私有真值拼入教师 prompt。

4 条不同 Train 资产均以预算1、最多3次真实请求、每决策最多2次 schema 尝试串行运行；实际 **8次请求、13,603 token，4/4 合法终止，3/4 独立审核通过**。通过项是 clean broad-scope `uncertain`、front 未暴露 offset 的 `uncertain`、以及静态 front 未暴露 joint-axis 的 `uncertain`。明确 front-exposed offset `PM_10697` 仍被 Kimi 判为 `uncertain`，审核以 `uncertain_misses_exposed_objective_defect` 拒绝；这证明审核没有把“合法 JSON”混同为“可信诊断”。本批没有合格 `fail` 或 `pass`。

`scripts/export_interactive_teacher.py` 仅导出 pipeline integrity 与独立证据审核同时通过的 Episode，保留逐决策可见 prompt、图片与 SHA、动作/终止目标、来源身份、split、源 Episode/Audit SHA 和审核结论；拒绝原因写入 `rejected.jsonl`。`kimi-round2-sft-export-v2` 含 **3个 Episode、3条 action、3条 terminal**，现有 formal Episode loader 实际加载6行/3个 terminal；本地 Qwen3-VL-2B processor 进一步成功生成真实图像张量与 assistant-only mask（1,134 input tokens、75 supervised tokens）。因此 API→Chromium→合法终止→独立审核→JSONL→loader/processor 的接口链路已经打通，但样本仅适合作为 `uncertain`/停止与动作监督的小型交付物，不能据此扩量或声称缺陷诊断能力。

完整计数、逐案例 SHA、拒绝原因和本地 artifact 路径见 [`kimi-interactive-teacher-round2.json`](evidence/kimi-interactive-teacher-round2.json)。复现单条采集时替换 `--sample-id` 与新 `--out`；已付费目录不得 resume 或覆盖：

测试方面，非浏览器套件为223 passed、2 skipped；交互教师与 verifier 的定向 Chromium 测试分别1/1与4/4通过。完整 browser suite 为19 passed、1 failed，唯一失败是既有 `test_catalog_clean_and_mixed_are_shard_independent` 对随机生成结果必须含双 corruption 的断言；本轮未修改该 generator/config/test 路径，未为追求全绿扩大修复范围。

```bash
python -m scripts.run_interactive_teacher --mode kimi \
  --sample-id b124e209888e8e2e692ad38a --budget 1 \
  --max-real-requests 3 --decision-attempts 2 --estimated-input-tokens 8192 \
  --third-party-transmission-reviewed \
  --transmission-review-basis partnet-approved-noncommercial-research-terms-no-explicit-render-api-prohibition \
  --out output/interactive-teacher/kimi-round2-insufficient-axis-pm102080
python -m scripts.export_interactive_teacher \
  --episode output/interactive-teacher/kimi-round2-clean-pm101870 \
  --episode output/interactive-teacher/kimi-round2-observable-offset-pm101714 \
  --episode output/interactive-teacher/kimi-round2-observable-offset-pm10697 \
  --episode output/interactive-teacher/kimi-round2-insufficient-axis-pm102080 \
  --out output/interactive-teacher/kimi-round2-sft-export-v2
```

### 第三轮 joint_axis 运动取证与诊断（2026-10-15）

第二轮记录、目录与导出保持不变，作为静态参考基线；它们早于本轮的冻结 witness 清单，**不满足第三轮审核标准，不应与第三轮数据一并交付**。

**审核修复。** `turnitover/teacher/witnesses.py` 在任何教师调用前冻结一份清单：对 front/oblique_fr 的全部运动 witness 重新渲染候选，并与 v4 冻结像素逐一核对；每个 witness 记录 Chromium 实际使用的、经限位截断后的关节值。`zero` 是下限，`rest` 是值 0 截断到限位后的姿态，两者只有在下限为 0 时才重合；审核按实际状态匹配，不信任 detent 名称（旧实现会把 `rest` 当作 front 静止 witness 的同义词）。`turnitover/teacher/evidence.py` 把每张实时观察图按（视角、实际关节值、像素 SHA）绑定到冻结 witness；只有同时“绑定成功”且“与某个可见参考状态可比”的观察才计为证据。Gold、v4 annotation、清单、Episode 四者须在 id、asset、split_group、候选/参考 SHA 和 Gold/annotation 行 SHA 上一致，split 必须是 Train/Validation，且行不得出现在冻结 Test 中；prompt 会扫描 Gold 文本、私有参数、程序源码与程序 SHA。审核分别报告动作合法性、每步是否带来新的客观证据、证据是否充分、诊断结果（`correct_fail`、`incorrect_fail`、`fail_without_sufficient_evidence`、`uncertain_after_sufficient_evidence`、`trustworthy_uncertain` 等）。导出会在审核后 Episode 被修改时拒绝；动作行按与标签无关的规则入选（合法、绑定、可比、新状态），缺陷相关性只作为元数据保留，避免隐藏 Gold 决定哪些可见前缀进入监督。

**案例与参考协议。** 选取4个 checker 确认的单缺陷 `kinematics.joint_axis` Train 行（PM_102080、PM_5088、PM_103275、PM_25308），每个配同资产 clean 对照。清单证明这4个缺陷在静态 front 下都不暴露，因此静态参考不足以诊断。本轮使用明确标注、与标签无关的 `front_motion_v1` 参考：front 视角下 rest 加上每个关节在 grid 中的非 rest detent，每张图只驱动一个关节。它独立于 round2 的 `static_front`，学生看不到 Gold、标签或参考程序参数。

**两组真实 API 实验（k3-256k，串行，每决策最多2次 schema 尝试）。** 固定组由协议先执行 front rest → 各关节 detent（换关节前显式回到 rest），并在任何付费调用前完成绑定 dry run；之后只请求一次结构化诊断，固定步骤不作为动作监督导出。主动组的预算等于固定协议长度，所有动作都由 Kimi 根据可见历史自主选择，可以提前停止。4次网关超时或空响应各用新目录重跑一次（超时 300 s），失败目录保留但不导出。

| 指标 | 固定充分证据 | 主动取证 |
|---|---:|---:|
| Episode（axis + clean） | 8 (4+4) | 8 (4+4) |
| 动作与终判均合法 | 8 | 8 |
| axis 获得充分且可比的运动证据 | 4/4（协议给定） | **4/4（自主获取）** |
| axis 严格正确（类型+关节+部件+证据步） | 1/4 | 1/4 |
| axis 类型+关节正确 | 2/4 | 3/4 |
| 充分证据后仍 `uncertain` | 1 | 1 |
| 证据不足却猜 `fail` | 0 | 0 |
| clean 可信 `uncertain` / 误报 fail / 无依据 pass | 3 / 0 / 1 | 4 / 0 / 0 |
| 合法 + 证据充分 + 诊断正确 | **1** | **1** |
| 独立审核通过 | 4 | 5 |

主动组的4个 axis 案例都先请求 front rest，再驱动目标关节；首次新证据分别出现在第1、1、4、4步（双关节资产先探索了 joint_0，复位后才驱动 joint_1）。3条 Episode 在预算内主动停止。PM_5088 在两组中都得到完全正确的 `fail`（joint_axis / joint_0 / link_0，证据步均暴露缺陷）。失败归因：PM_102080 的棱柱关节轴误差在 front 正交视图中只改变锅盖边缘的明暗（最大1.1%像素变化），两组都输出 `uncertain`。审核按冻结标准仍拒绝，但证据在感知上很弱，主要归因于“仅 front 的运动参考显著性不足”。PM_103275 两组都定对了类型和关节，但部件多报了 joint_1 的静止父件 link_2；关节→部件映射不在可见 context 中（只能通过 `query_runtime hierarchy` 获得，Kimi 从未使用；固定组中根本没有该信息），属于定位错误加上可见结构信息不足。PM_25308 固定组把出平面的抽屉运动诊断为 `joint_range`（front 投影中位移变短，属于诊断错误），主动组定对了类型和关节，但部件选成父件 link_2。

**导出。** 采用每个（模式, 样本）的最终尝试，共16条。`kimi-round3-sft-export` 含 **32条动作行**（16条 Episode 均未被阻断；5条重复/复位或 runtime 动作被拒）与 **9条诊断行**（2条 `correct_fail`，7条 clean `uncertain`）。现有 formal loader 实际加载 actions 32 / diagnosis 9 / episodes 41 行，Qwen3-VL-2B processor 为 actions 和 diagnosis 各生成真实图像张量与 assistant-only mask。

**结论。** Kimi 能在合法、可审计的动作空间内自主获取诊断 joint_axis 所需的运动证据（4/4），取证不是瓶颈。拿到充分证据后，严格诊断正确率在两组中都只有1/4，类型+关节可达2–3/4，主要短板是部件定位和低显著性证据。这批数据可以作为谨慎停止和取证动作的小型接口交付给 `laq/sft-training`，但其中只有2条（同一资产）正确 `fail` 诊断，不足以支持缺陷诊断训练或能力结论。完整逐步轨迹、SHA、拒绝原因与复现命令见 [`kimi-interactive-teacher-round3-axis.json`](evidence/kimi-interactive-teacher-round3-axis.json)。

```bash
python -m scripts.freeze_teacher_witnesses --out output/interactive-teacher/axis-round3-frozen-witnesses \
  --sample-id b124e209888e8e2e692ad38a --sample-id 87873dda7b4e005dd094bb8e  # 其余6个见 axis-round3-cases.txt
python -m scripts.run_interactive_teacher --mode kimi --sample-id 78daaf49b71ff5f5df8a1a29 \
  --experiment-mode active --reference-protocol front_motion_v1 \
  --frozen-witnesses output/interactive-teacher/axis-round3-frozen-witnesses --budget 4 \
  --max-real-requests 7 --decision-attempts 2 --estimated-input-tokens 16384 \
  --third-party-transmission-reviewed \
  --transmission-review-basis partnet-approved-noncommercial-research-terms-no-explicit-render-api-prohibition \
  --out output/interactive-teacher/kimi-round3-active-pm5088-axis-new
python -m scripts.reaudit_interactive_teacher --episode output/interactive-teacher/kimi-round3-active-pm5088-axis-r2
HF_HUB_OFFLINE=1 ../TurnItOver/.venv-qwen/bin/python -m scripts.check_teacher_export_loading \
  --dataset output/interactive-teacher/kimi-round3-sft-export --out <new>.json \
  --model ../TurnItOver/models/Qwen3-VL-2B-Instruct
```

## 复现命令

以下命令要求已获官方访问权、已构建 `web/dist`，且各输出目录不存在：

```bash
python -m scripts.download_partnet_mobility \
  --out data/partnet-mobility-pilot --count 160 --seed 20261013 \
  --max-archive-mb 5 --workers 12

python -m scripts.convert_partnet_mobility \
  --root data/partnet-mobility-pilot \
  --out output/partnet-mobility-pilot-candidates-v3 \
  --test-plan configs/uniphys-test.json --test-root data/assets/uniphys/data

review_ids=$(jq -r '.assets[].asset_id' configs/partnet-mobility-pilot-v1.json | paste -sd, -)
python -m scripts.audit_catalog_motion \
  --catalog output/partnet-mobility-pilot-candidates-v3/catalog.json \
  --ids "$review_ids" --out output/partnet-mobility-motion-audit-v1

python -m scripts.convert_partnet_mobility \
  --root data/partnet-mobility-pilot \
  --out data/assets/partnet-mobility-trainval-v1 \
  --test-plan configs/uniphys-test.json --test-root data/assets/uniphys/data \
  --motion-reviews configs/partnet-mobility-pilot-v1.json

python -m turnitover generate \
  --config configs/generate_partnet_mobility_supervision.yaml \
  --out data/benchmarks/partnet-mobility-formal-v1/raw
python -m turnitover prepare-dataset \
  --shards data/benchmarks/partnet-mobility-formal-v1/raw/shard-0000.tar \
  --splits data/assets/partnet-mobility-trainval-v1/splits.json \
  --out data/benchmarks/partnet-mobility-formal-v1/prepared
python -m scripts.run_oracle_supervision \
  --benchmark data/benchmarks/partnet-mobility-formal-v1 \
  --config configs/generate_partnet_mobility_supervision.yaml \
  --out data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 \
  --episode-objective first_failure
python -m scripts.audit_training_dataset_expansion
```

审核配置中的人工结论不能由 smoke 命令自动重建；新批次必须重新看三视图/端点证据并形成新的审核清单，不能复制本批结论。

## 已知限制与下一阶段成本

- 数据许可限制为非商业研究/教育；下游发布前还需复核是否允许分发转换后的网格，默认只发布脚本、ID、哈希和审计，不重新分发上游资产。
- 21 类仍不等于类别泛化；Validation 7 个对象较小，混合关节仅 2 个且都在 Validation。下一批应优先补 Train 混合关节和本轮未覆盖类别。
- 当前缺陷只有 part offset 与 joint axis；clean 的“无配置缺陷”不证明现实中所有属性都正确，特权 Episode 的 47 个预算耗尽样本也不能标作 pass。
- 35 个转换 catalog 约 420 MiB；最终 raw、prepared、Episode 和审计合计约 1.1 GiB。首跑及失败导出在干净提交重跑通过后移入系统回收站，最终目录不混入旧版本。

按本次漏斗，160 个下载得到 94 个结构候选；优先审核的 48 个中 35 个通过。若后续仍维持相近分布，新增 100 个合格物体约需下载 **230–250 个**、浏览器与人工审核约 **140 个**结构候选；由于本次审核是优先抽样，这只是规划下限，不是通过率承诺。干净提交上140条真实渲染与 checker 生成耗时10分32秒，即当前硬件约4.5秒/样本；按每物体4条监督样本，100个新物体的生成阶段约30分钟，Episode 的教师渲染仍是另一项主要成本。下一批先扩到 80–100 个合格原物体并观察通过率漂移，再决定是否全量扫描；GAPartNet 只在取得官方完整包和许可记录后作为类别/部件标注补充。
