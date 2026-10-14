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

`oracle-first-failure-v3/` 为无模型、特权 counterfactual 教师的当前 Episode 导出：覆盖 138 个通过样本，Episode 行 Train/Validation 为 429/110；91 条因证据充分停止，47 条预算耗尽。全部保存图像均通过独立重渲染逐像素复核。该教师读取私有真值以选动作；下述就绪性验证发现 4 组相同可见前缀对应不同动作，因此这些动作标签明确不进入 SFT，**不是模型能力、人工 Gold 或可观测最优策略证明**。

自然生成错误单独登记在 [`partnet-mobility-natural-errors-v1.json`](../configs/partnet-mobility-natural-errors-v1.json)。本轮没有启动付费生成调用，故数量明确为 0、状态为 `not_collected`；没有用 corruption 注入标签冒充自然错误或独立人工 Gold。后续只有在人工不读取注入标签、依据真实渲染独立标注后，才可进入该集合。

机器可读总审计位于 `data/benchmarks/partnet-mobility-formal-v1/dataset-audit.json`，由 [`audit_training_dataset_expansion.py`](../scripts/audit_training_dataset_expansion.py) 从 catalog、split、raw/prepared manifest 与 Episode 重新计算，不依赖本文手填统计。

## SFT 数据就绪性验证

### 数据版本与输入协议

本次只验证数据进入既有 Qwen3-VL-2B 管线，不创建优化器、不反向传播、不保存 checkpoint。审计对象为 PartNet `oracle-first-failure-v3`，manifest SHA-256 为 `f95302809fa135d2d161577dee9b44ff405a671287ae07bc1015f40a8e1f546b`；Qwen 配置 SHA-256 为 `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b`。正式资产池仍为 PartNet 35 个加 UniPhys 5 个，共 Train 31 / Validation 9；当前数据工作树不复制被 Git 忽略的 UniPhys 大资产，联合身份审计读取相邻工作树中同一冻结 catalog，并把 catalog/splits SHA 写入报告。

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

机器可复核报告为 `output/partnet-mobility-sft-readiness-v1.json`，SHA-256 `256c83ad164a74034a6fde0d8eccd7c220a02dcb5e34aaf6b046f523490d280a`。结论是：**当前 138 条数据已达到“结构化诊断与停止判断”正式 SFT 的管线就绪标准**；不支持动作 SFT、`pass` 校准或泛化结论，正式训练前应把上述偏置作为固定实验边界。

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

两组 110 条样本的用户消息、图像、历史和 terminal 前缀逐条完全相同，差异只在 assistant verdict。普通组用独立 corruption Gold 构造完整诊断，共 111 个 finding；证据组只保留当前公开观察下通过既有 exposure 门禁的 74 个 finding。机器审计 `output/partnet-qwen3vl2b-supervision-audit-v1.json` 逐 ID 列出 **37 条普通组独有的 finding，全部为 `kinematics.joint_axis`**；它们是合法 Gold，但超出冻结可见证据。两组都不读取未来观察或把 annotation/Oracle 信号放进模型输入，不训练401条特权动作，也不把 hidden-clean 变成 `pass`。没有使用不同 Episode 前缀或另造停止标签。

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

status 的满分存在明确捷径：19个 `fail` Validation 例都在第1/2步因 `evidence_sufficient` 形成 terminal，9个 `uncertain` 都在第6步预算耗尽，而 `history/remaining` 是合法公开输入。因此两组的 100% status/macro-F1 只能说明学会当前 terminal 数据分布，**不能说明学会视觉提前停止**。本轮只在强制 terminal 前缀要求最终 verdict，所谓 terminal stop=28/28 等价于合法输出率；没有非 terminal 停止标签，也未评测提前停止。

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

### 结论、产物与复现

已证实：（1）terminal status 与 step/remaining 完全关联；（2）两组 SFT 在同预算前缀上均不区分证据充分性；（3）删除 `remaining` 会显著破坏两组 status，证据组尤其从9/9 terminal uncertain 全部转为 fail；（4）证据监督真实减少了 axis FP/unsupported，但以 offset TP 与总体 recall 下降为代价。尚不能识别：模型是否真正识别视觉证据、`remaining` 相对其他阶段线索的独立效应，以及缺席的2个正式 Validation 物体上的泛化。

逐前缀预测与配置位于 `output/partnet-qwen3vl2b-diagnostic-{base,ordinary,evidence}-prefix-v1/`，预算消融位于对应 `-no-remaining-v1/`；机器汇总为 `output/partnet-qwen3vl2b-diagnostic-analysis-v1.json`，SHA-256 `4f968468be00586c7daa6da791027c3a56a7610a1399b33410a5d2491fcd1ed0`。报告包含逐物体、缺陷组合、可见 finding 类型、配对差、bootstrap 及全部输入预测 SHA；冻结 Part 2 Test 未读取，Gold/证据标准未修改。

实际命令（普通/证据模型分别增加对应 `--adapter output/partnet-qwen3vl2b-{ordinary,evidence}-sft-v1/checkpoint`）：

```bash
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/evaluate_qwen_episode_diagnostics.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --out output/partnet-qwen3vl2b-diagnostic-base-prefix-v1 --condition causal_prefixes --max-new-tokens 512
CUDA_VISIBLE_DEVICES=0 ../TurnItOver/.venv-qwen/bin/python scripts/evaluate_qwen_episode_diagnostics.py --dataset data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v3 --model ../TurnItOver/models/Qwen3-VL-2B-Instruct --out output/partnet-qwen3vl2b-diagnostic-base-no-remaining-v1 --condition terminal_without_remaining --max-new-tokens 512

python scripts/analyze_qwen_episode_diagnostics.py \
  --base-prefix output/partnet-qwen3vl2b-diagnostic-base-prefix-v1/predictions.jsonl \
  --ordinary-prefix output/partnet-qwen3vl2b-diagnostic-ordinary-prefix-v1/predictions.jsonl \
  --evidence-prefix output/partnet-qwen3vl2b-diagnostic-evidence-prefix-v1/predictions.jsonl \
  --base-no-remaining output/partnet-qwen3vl2b-diagnostic-base-no-remaining-v1/predictions.jsonl \
  --ordinary-no-remaining output/partnet-qwen3vl2b-diagnostic-ordinary-no-remaining-v1/predictions.jsonl \
  --evidence-no-remaining output/partnet-qwen3vl2b-diagnostic-evidence-no-remaining-v1/predictions.jsonl \
  --out output/partnet-qwen3vl2b-diagnostic-analysis-v1.json --seed 20261013 --bootstrap-samples 10000
```

下一轮最值得做的是构造**固定 horizon、阶段线索匹配的视觉证据对照**：在相同 step、remaining、历史长度、图像数和动作类型下，收集证据充分/不充分且跨物体平衡的真实前缀；评测时同时提供原图与不改变合法历史的图像内容消融，训练时再用这些固定 horizon 的可见证据 verdict 做多 seed 对照。现有自然前缀只能匹配预算阶段，不能匹配历史内容，先补这个评测设计比直接扩大当前 terminal SFT 更能回答因果问题。

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
