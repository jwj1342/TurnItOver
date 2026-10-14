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

## 第二轮 v4 监督训练与受控评测

### 训练设置与 checkpoint

2026-10-14 已在两张 RTX 4090D 上从同一 Qwen3-VL-2B-Instruct 重新训练普通/证据约束 LoRA。两组都用 v4 Train 的110条 terminal 诊断和429条逐前缀停止监督，共539样本、3 epochs、1617 optimizer steps；seed `20261013`、AdamW `2e-4`、rank 8、alpha 16、dropout 0、`q/k/v/o_proj` 完全一致。停止样本权重为 `110/429`，诊断为1，使每个 epoch 两类目标的标称总权重相等；每步日志分别记录任务、assistant-token mean CE、权重和加权 loss。普通组的110条诊断包含完整 annotation finding，证据组只保留当前可见前缀支持的 finding；429条停止目标和所有公开输入逐条相同。私有 Gold 只用于普通目标构造和离线评分，不进输入；证据目标不含隐藏缺陷、未来观察或未暴露错误，Oracle 动作不训练，预算耗尽只标 `stop/uncertain`。

| 训练 | 样本 / 步数 | diagnosis raw loss mean | stopping raw loss mean | adapter SHA-256 |
| --- | ---: | ---: | ---: | --- |
| 普通 v4 | 539 / 1617 | .230 | .0261 | `9b77809f61156769e5074718cc58c74ef8d6b517d78f58a153158a9637b11e2b` |
| 证据 v4 | 539 / 1617 | .213 | .0248 | `c1b7f3b1171cb94a180b11854b0c07ccfc7c1e5c7c827eb25773df5457e42dbd` |

两份 `run.json` 均为 `complete`，checkpoint、逐步曲线、配置和 processor 位于 `output/partnet-qwen3vl2b-{ordinary,evidence}-sft-v4-round2/`。训练时 CUDA 对 CuBLAS/Flash Attention 发出 warn-only 非确定性警告，因此配置、seed 和顺序可复现，但不声称权重逐比特复现。loss 只用于训练完整性审计，不作为效果结论。

### 留出诊断与停止结果

评测实际覆盖28条 Validation Episode、110个因果前缀和7个独立 PartNet 物体（`PM_100202/102252/102590/10797/1280/34610/45759`）；正式 Validation 池另2个 UniPhys 物体没有 Episode。terminal 诊断使用相同图像、greedy decode 和 `max_new_tokens=512`：

| 指标 | 未微调底座 | 普通 v4 | 证据 v4 | 无图像预算基线 |
| --- | ---: | ---: | ---: | ---: |
| status accuracy | 7/28 | 28/28 | 28/28 | 28/28 |
| exact verdict + 定位 | 7/28 | 17/28 | 17/28 | 23/28 类型 exact；不定位部件 |
| finding TP / FP / FN | 0 / 0 / 19 | 8 / 11 / 11 | 8 / 11 / 11 | 0 / 0 / 19 |
| finding F1 | 0 | .421 | .421 | 0 |
| defect-type F1 | 0 | .737 | .842 | .737 |
| part/joint localization F1 | 0 | .421 | .421 | 不可测 |
| unsupported / predicted | 0 / 0 | 15 / 19 | 11 / 19 | 0 / 0 |

证据组相对普通组少4个 unsupported finding，defect-type F1 高 `.105`，主要是 axis FP 5→3；但 exact、精确 finding、部件定位以及 TP/FP/FN 完全相同，offset TP/FP/FN 为4/6/11→4/8/11，未显示漏检减少。两组的 `fail/uncertain` 都是19/9全对，但无图像基线也为28/28，故 status 仍被预算阶段完全解释，不能作为视觉收益。

显式停止评测要求模型只返回 `stop_decision`，`max_new_tokens=128`：

| 条件 | 合法输出 | decision accuracy | 提前 stop 前缀 | terminal 漏停 | 首停 exact / early / missing | 预算耗尽 uncertain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 底座 | 0/110 | 0/110 | 0 | 28 | 0 / 0 / 28 | 0/9 |
| 普通 v4 | 110/110 | 84/110 | 22 | 4 | 10 / 18 / 0 | 9/9 |
| 证据 v4 | 110/110 | 80/110 | 27 | 3 | 10 / 18 / 0 | 9/9 |
| 无图像预算基线 | n/a | 91/110 | 0 | 19 | 9 / 0 / 19 | 9/9 |

逐前缀监督让两组学会了输出 schema 和预算耗尽的 `stop/uncertain`，但没有改善正确停止时机。step 1 / remaining 5 时，有证据10条与无证据18条，两组都100% stop；step 2 / remaining 4 时，普通组 stop rate 为有证据5/9、无证据4/9，证据组反而为有证据6/9、无证据9/9。证据组总体准确率低于普通组及无图像基线，提前 stop 更多，首次停止仍与第一轮相同为10 exact / 18 early。因此本轮不能回答“证据约束模型更准确利用视觉证据”为是，逐前缀停止能力也没有真正改善。

机器审计为 `output/partnet-qwen3vl2b-round2-audit-v4.json`（SHA-256 `540072fd61c038f62edf56604fa5d1a3e67d9414313ace892b0facab3f5f63dd`）；监督审计 SHA-256 仍为 `2ccbdc…cd176`。逐案例预测分别在 `*-terminal-eval-v4-round2/` 与 `*-stopping-eval-v4-round2/`，训练曲线和 checkpoint 在 `*-sft-v4-round2/`。一次底座停止评测完成推理后因分层聚合器错误失败，原目录保留为 `partnet-qwen3vl2b-base-stopping-eval-v4-round2-failed-aggregate/`；修复单测后正式重跑完成。冻结 Part 2 Test 未读取或修改，未接入 API，未训练特权动作。

```bash
DATA=/91004t2/laq/code/ThreejsJudge/TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4
MODEL=../TurnItOver/models/Qwen3-VL-2B-Instruct
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python -m scripts.train_qwen_episode_sft --dataset "$DATA" --model "$MODEL" --out output/partnet-qwen3vl2b-ordinary-sft-v4-round2 --supervision ordinary --seed 20261013 --epochs 3 --learning-rate 0.0002 --lora-rank 8 --lora-alpha 16
CUDA_VISIBLE_DEVICES=1 ../TurnItOver/.venv-qwen/bin/python -m scripts.train_qwen_episode_sft --dataset "$DATA" --model "$MODEL" --out output/partnet-qwen3vl2b-evidence-sft-v4-round2 --supervision evidence --seed 20261013 --epochs 3 --learning-rate 0.0002 --lora-rank 8 --lora-alpha 16
python -m scripts.evaluate_budget_baseline --dataset "$DATA" --out output/partnet-no-image-budget-baseline-v4-round2
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python -m scripts.evaluate_qwen_episode_sft --dataset "$DATA" --model "$MODEL" --out output/partnet-qwen3vl2b-base-terminal-eval-v4-round2 --max-new-tokens 512
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python -m scripts.evaluate_qwen_episode_sft --dataset "$DATA" --model "$MODEL" --adapter output/partnet-qwen3vl2b-ordinary-sft-v4-round2/checkpoint --out output/partnet-qwen3vl2b-ordinary-terminal-eval-v4-round2 --max-new-tokens 512
CUDA_VISIBLE_DEVICES=1 ../TurnItOver/.venv-qwen/bin/python -m scripts.evaluate_qwen_episode_sft --dataset "$DATA" --model "$MODEL" --adapter output/partnet-qwen3vl2b-evidence-sft-v4-round2/checkpoint --out output/partnet-qwen3vl2b-evidence-terminal-eval-v4-round2 --max-new-tokens 512
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python -m scripts.evaluate_qwen_stopping --dataset "$DATA" --model "$MODEL" --out output/partnet-qwen3vl2b-base-stopping-eval-v4-round2 --max-new-tokens 128
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python -m scripts.evaluate_qwen_stopping --dataset "$DATA" --model "$MODEL" --adapter output/partnet-qwen3vl2b-ordinary-sft-v4-round2/checkpoint --out output/partnet-qwen3vl2b-ordinary-stopping-eval-v4-round2 --max-new-tokens 128
CUDA_VISIBLE_DEVICES=1 ../TurnItOver/.venv-qwen/bin/python -m scripts.evaluate_qwen_stopping --dataset "$DATA" --model "$MODEL" --adapter output/partnet-qwen3vl2b-evidence-sft-v4-round2/checkpoint --out output/partnet-qwen3vl2b-evidence-stopping-eval-v4-round2 --max-new-tokens 128
python scripts/compare_qwen_sft.py --base output/partnet-qwen3vl2b-base-terminal-eval-v4-round2/metrics.json --ordinary output/partnet-qwen3vl2b-ordinary-terminal-eval-v4-round2/metrics.json --evidence output/partnet-qwen3vl2b-evidence-terminal-eval-v4-round2/metrics.json --base-stopping output/partnet-qwen3vl2b-base-stopping-eval-v4-round2/metrics.json --ordinary-stopping output/partnet-qwen3vl2b-ordinary-stopping-eval-v4-round2/metrics.json --evidence-stopping output/partnet-qwen3vl2b-evidence-stopping-eval-v4-round2/metrics.json --budget-baseline output/partnet-no-image-budget-baseline-v4-round2/metrics.json --ordinary-train output/partnet-qwen3vl2b-ordinary-sft-v4-round2/run.json --evidence-train output/partnet-qwen3vl2b-evidence-sft-v4-round2/run.json --out output/partnet-qwen3vl2b-round2-audit-v4.json
```

下一轮必须增加阶段匹配的合法图像反事实对：保持 step、remaining、历史结构和动作类型不变，只改变真实可见证据内容；并补多 seed、可信 pass、自然错误、更多独立物体及单独冻结的 `comprehensive_within_budget`。当前自然前缀不是图像反事实配对，因而即便某一指标改善，也不能排除物体、动作历史或阶段捷径。

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

## v5：证据驱动交互决策数据（2026-10-14）

**已经实际生成并审核依赖可见证据、不能由预算阶段决定的停止训练样本。停止专项数据可用，但完整第三轮多任务SFT尚未就绪，本轮未训练。** 资产和v4数据只读复用 `../TurnItOver-dataset`，浏览器依赖只读复用 `../TurnItOver`；所有新数据位于当前worktree。未调用付费API、未启动或停止另一worktree的Kimi采集、未修改冻结Part 2 Test，未新建文档。

### 实际轨迹与标签来源

主产物为 `data/partnet-motion-decisions-v5/`。[`build_visual_decisions.py`](../scripts/build_visual_decisions.py) 对原有110 Train / 28 Validation候选执行三种固定顺序，每条六步，共414条真实轨迹、2,484次真实观察。每个物体只选择公开关节列表的第一个关节，取非零端点（优先upper，否则lower）；没有按Gold选择关节或动作。令F/L/T/B为front/left/top/back，M为驱动到端点，R为真实回零：三种顺序是 `F,M,L,R,T,B`、`M,L,R,F,T,B`、`L,M,F,R,T,B`。回零也消耗一步并保存真实观察，未重排旧图冒充执行。

学生预先获得八张真实参考图：六视角静止态，以及front/left下所选关节端点态。`context.reference_states` 明确各图的视角与关节状态，历史记录当前观察对应的参考索引。**这些都是公开任务输入，不是事后借用学生不可见的参考运动。** 相机、灯光、framing和其余关节状态相同；每个前缀保存实际PNG/SHA、动作、实际关节值或相机、可见历史、证据步骤、差异bbox和停止依据。消息里的图像名使用位置别名，避免候选ID/路径捷径。

[`visual_decisions.py`](../turnitover/oracle/visual_decisions.py) 的标签函数只接收对应参考PNG、已观察PNG和预算；不接收Gold、程序、exposure表或未来候选图像。轮廓差异须同时达到64像素、物体投影并集的1%、最大连通区域16像素，才标作可定位的姿态形状不一致证据。没有充分证据时，只有当前所有观察与对应参考逐像素相同才标 `continue`；其余不明确差异隔离。预算归零且没有证据为 `stop/uncertain`，不标pass、不猜缺陷类型或部件定位。Gold/corruption仅用于重建既有候选并核验SHA，不参与这些标签决策。这是保守的程序化视觉审核，不等于独立人工缺陷Gold。

采集策略执行完整六步，因此也包含已有充分证据后的离线探索前缀；它们合法，但不冒充最优first-failure策略。预算剩余时的证据停止标 `termination_kind=active`；第六步一律 `forced`，结果可为fail或uncertain。另导出 `*.first-stop.stopping.jsonl`，去除首次充分证据之后的延续前缀。

### 全量审核与数据规模

[`audit_visual_decisions.py`](../scripts/audit_visual_decisions.py) 从PNG重算全部标签/bbox和因果历史，逐字段核对输出，复用生产 `load_examples` 检查身份、图像闭合与SHA；从动作重建参考状态，检查跨顺序同候选同状态的真实PNG一致性。2,484个前缀中隔离43个（Train22 / Validation21），去掉434条完全相同的公开文本及图像内容序列，保留2,007条。`prefix-aliases.audit.jsonl` 保留原始轨迹前缀到去重训练行的映射，避免丢失同候选对照来源，也不将重复输入计作新增样本。

| 主数据，去重后 | Train | Validation |
| --- | ---: | ---: |
| 原物体 / 原候选 / 真实轨迹 | 28 / 110 / 330 | 7 / 28 / 84 |
| 决策行 | 1,618 | 389 |
| stop/fail / continue / stop/uncertain | 1,164 / 379 / 75 | 263 / 105 / 21 |
| 主动停止 / 预算强制终止 | 963 / 276 | 215 / 69 |
| 首次充分证据后的延续前缀 | 963 | 215 |
| 去除上述延续后的首停子集 | 655 | 174 |
| 同物体、同动作等公开字段匹配对 | 359 | 102 |
| 上述对照涉及物体 / 行数 | 26 / 718 | 7 / 204 |
| 首停子集中的严格匹配对 | 97 | 31 |
| 同候选、同阶段、不同观察顺序的正反分组 | 10 | 3 |
| 上述不同顺序分组涉及原物体 | 10 | 2 |

两类对照各有独立清单：

- `*.pairs.jsonl`：同原物体、八张参考SHA、完整公开context、动作序列、相机/关节值、步数、预算及图像数均相同；两侧来自分别实际执行的不同候选，只有观察像素不同，一侧stop/fail、一侧continue。每行最多参加一对。Train在step1–5为72/71/72/72/72对，Validation为18/21/21/21/21对。`validation.matched.stopping.jsonl` 是冻结的204行同预算评测集，仅使用原有7个留出物体。
- `*.order-pairs.jsonl`：同一真实候选和相同预算，仅动作顺序不同；Train10组均在step1，Validation在step1有2组、step2有1组。清单同时给出原始前缀ID和去重后训练行ID，供沿真实轨迹复核。它本身允许动作历史不同，因此还需要上一类严格匹配对排除动作阶段捷径。

Train/Validation物体组与完整公开输入交集均为0；相同公开输入的标签冲突为0。v4的该冲突指标也为0，不宣称此项进一步提升。对12个具有step1同候选正反对照的原物体逐图进行了助手目视复核，`order-review-0.png` 至 `order-review-2.png` 显示一致的静止/侧视状态与明显不一致的运动状态；这不是全体正例独立人工Gold。阈值未根据Validation调优。

### 相对v4的实际变化与覆盖缺口

v4按同一严格匹配原则可找到Train38 / Validation9对，仅覆盖step1/2；v5主数据扩展到359 / 102对及step1–5。首停子集仍有97 / 31对，不能把大量已有证据后的延续前缀都算成新取证事件。完整轨迹首次充分证据的Train分布为step1/2/3：164/32/5，另外129条无充分证据；Validation为step1/2/3/5：35/7/3/3，另外36条无充分证据。真正较晚首次获得证据仍有限，同候选顺序敏感对照只有10个Train、2个Validation物体。

仅用step/remaining/image_count在Train拟合多数查找，实际Validation为v4 `91/110=82.73%`、v5 `263/389=67.61%`。严格匹配集内所有非图像公开字段相同而标签相反，故任意仅依赖这些字段的确定性分类器上限为50%。这证明配对标签不能由预算或动作字段决定，**不证明模型已经学会视觉停止**。v5公开参考图更多、任务限定为匹配姿态形状一致性，和v4不是等输入等任务的效果实验。

此前还完成了静态六参考图试点 `data/partnet-visual-decisions-v5/`：414条真实轨迹、2,484次观察，隔离88个模糊前缀、去重581条后得到Train1,455 / Validation360条和373 / 97个严格匹配对。它保留供审核，不与主数据直接相加宣称独立样本或物体扩量；主数据补入真实关节观察以覆盖仅改变顺序的取证差异。两套数据总计约102 MiB，均复用同一35个原物体。

### 下一步动作审核与训练条件

[`motion_decisions.py`](../turnitover/oracle/motion_decisions.py) 逐前缀枚举公开视角与所选关节的回零/端点动作，记录合法性、对应参考是否存在、新状态覆盖和eligible；重复已观察状态不作为有效下一步，预算归零时不可用。未观察图像的信息增益保持 `unknown_without_execution`；`action-outcomes.audit.jsonl` 另记录真实执行后是否首次出现充分证据及新状态覆盖，只用于审核，不倒灌学生标签或构造特权动作目标。未覆盖其余关节、runtime查询或最优动作监督。

真实本地Qwen3-VL processor已检查全部1,618/389行：Train token长度1,218–2,441，Validation1,231–2,447，assistant监督35–40 token；多图输入、前缀遮罩与非空监督全部通过。未加载模型权重、未前向或训练。30项相关单测通过。

**第三轮条件：停止专项数据已就绪，完整多任务训练尚未就绪。** 可用标准 `load_examples(root, split, 'stopping', require_formal_training=True)` 消费；优先考虑首停子集及严格配对集。现有训练入口强制同时提供diagnosis与stopping，本轮没有伪造diagnosis满足入口，也未改动训练脚本。正式第三轮前须确定八参考姿态输入协议和停止专项任务接入方式；若目标仍是原单参考图全关节验证，则本数据不能直接替代。全关节、多步顺序敏感覆盖、可信pass、自然错误人工Gold及多seed效果仍缺失，不以本轮的轮廓差异门禁冒充这些能力。

### 复现与产物

输出目录须不存在；使用本地SwiftShader单浏览器顺序执行，无需网络/API：

```bash
export PLAYWRIGHT_BROWSERS_PATH=/91004t2/laq/code/ThreejsJudge/TurnItOver/.cache/ms-playwright
../TurnItOver/.venv-local/bin/python -m scripts.build_visual_decisions --motion \
  --benchmark ../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1 \
  --asset-workspace ../TurnItOver-dataset --harness-workspace ../TurnItOver \
  --out data/partnet-motion-decisions-v5

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 USE_TF=0 \
  ../TurnItOver/.venv-qwen/bin/python -m scripts.audit_visual_decisions \
  --dataset data/partnet-motion-decisions-v5 \
  --baseline ../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4 \
  --processor-model ../TurnItOver/models/Qwen3-VL-2B-Instruct

../TurnItOver/.venv-local/bin/python -m pytest \
  tests/unit/test_visual_decisions.py tests/unit/test_oracle_episodes.py \
  tests/unit/test_sft_diagnostics.py tests/unit/test_sft_readiness.py
```

静态试点去掉 `--motion`、更换输出目录即可复现；仅重导出标签可用 `--reexport <已完成目录>`，不重新生成图像。初次默认浏览器缓存版本不匹配的空失败目录为 `data/partnet-visual-decisions-v5-failed-browser/`，不计入正式数据。

主目录的 `manifest.json` 保留来源与标签代码SHA，`trajectories.jsonl` 保存真实观察，`audit.json` 记录统计、逐阶段分布、文件SHA及processor结果；`*.matched.stopping.jsonl` 是严格配对子集，`*.first-stop.stopping.jsonl` 是首停子集。代码、测试和现有文档本地提交，不push；数据与PNG按现有约定本地保留。

本次主数据冻结摘要：`audit.json` SHA-256为 `f9497dbf0dee2a1bd8b8b6ba3666c16fb3c62907eadb98681c5b74f57b9f3ecf`；`train.stopping.jsonl` 为 `10754f2a969e7f3f77adc629c47b9f7ce21d196c804ccca5b96fbf5e4fa3ca74`；独立204行 `validation.matched.stopping.jsonl` 为 `890fca8fc589d4cf343746cad4332ef65ca7237ec400a1a7d10f789b65055706`。其余文件SHA见审计报告。
