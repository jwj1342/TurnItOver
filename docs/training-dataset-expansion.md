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
