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

`oracle-first-failure-v2/` 为无模型、特权 counterfactual 教师的 Episode 导出：覆盖 138 个通过样本，得到 Train/Validation 动作例 660/168、Episode 行 429/110；91 条因证据充分停止，47 条预算耗尽。全部保存图像均通过独立重渲染逐像素复核。该教师读取私有真值以选动作，只能作为动作监督候选，**不是模型能力、人工 Gold 或可观测最优策略证明**。

自然生成错误单独登记在 [`partnet-mobility-natural-errors-v1.json`](../configs/partnet-mobility-natural-errors-v1.json)。本轮没有启动付费生成调用，故数量明确为 0、状态为 `not_collected`；没有用 corruption 注入标签冒充自然错误或独立人工 Gold。后续只有在人工不读取注入标签、依据真实渲染独立标注后，才可进入该集合。

机器可读总审计位于 `data/benchmarks/partnet-mobility-formal-v1/dataset-audit.json`，由 [`audit_training_dataset_expansion.py`](../scripts/audit_training_dataset_expansion.py) 从 catalog、split、raw/prepared manifest 与 Episode 重新计算，不依赖本文手填统计。

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
  --out data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v2 \
  --episode-objective first_failure
python -m scripts.audit_training_dataset_expansion
```

审核配置中的人工结论不能由 smoke 命令自动重建；新批次必须重新看三视图/端点证据并形成新的审核清单，不能复制本批结论。

## 已知限制与下一阶段成本

- 数据许可限制为非商业研究/教育；下游发布前还需复核是否允许分发转换后的网格，默认只发布脚本、ID、哈希和审计，不重新分发上游资产。
- 21 类仍不等于类别泛化；Validation 7 个对象较小，混合关节仅 2 个且都在 Validation。下一批应优先补 Train 混合关节和本轮未覆盖类别。
- 当前缺陷只有 part offset 与 joint axis；clean 的“无配置缺陷”不证明现实中所有属性都正确，特权 Episode 的 47 个预算耗尽样本也不能标作 pass。
- 35 个转换 catalog 约 420 MiB；本轮全套 benchmark（包含一次失败导出和 prepared 备份）约 1.6 GiB。正式归档前可在核对哈希后删除旧失败/备份副本，但本轮保留用于审计。

按本次漏斗，160 个下载得到 94 个结构候选；优先审核的 48 个中 35 个通过。若后续仍维持相近分布，新增 100 个合格物体约需下载 **230–250 个**、浏览器与人工审核约 **140 个**结构候选；由于本次审核是优先抽样，这只是规划下限，不是通过率承诺。140 条真实渲染与 checker 生成耗时约 15 分钟，即当前硬件约 6.4 秒/样本；按每物体 4 条监督样本，100 个新物体的生成阶段约 43 分钟，Episode 的 6,200 次教师渲染则是更主要成本。下一批先扩到 80–100 个合格原物体并观察通过率漂移，再决定是否全量扫描；GAPartNet 只在取得官方完整包和许可记录后作为类别/部件标注补充。
