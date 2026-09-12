# 数据生成、质量控制与评测准备

这轮实现补齐了多资产遍历、正常与组合样本、标签有效性检查、资产组划分、输入/真值隔离导出，
以及保存观测上的 runtime 基线评测。它还不是 VLM 训练系统。后续已补充[暴露教师与训练导出](oracle-supervision.md)及
[真实修复 pilot](repair-experiment.md)，LoRA 训练与完整 RP 上界仍未完成。资产选型与实际下载记录见 [资产调研](asset-sources.md)。

## 先运行数据质量试验

```bash
source scripts/setup_local.sh
python -m turnitover generate --config configs/generate_dataset.yaml --out data/runs/pilot-v2
python -m turnitover inspect data/runs/pilot-v2/shard-0000.tar --n 8
```

配置默认生成 16 个 toy 样本，25% 的正常样本采样概率，其余随机施加 1–2 种不同腐蚀。
比例是采样概率，不保证小样本批次达到精确配额。不能把这批同源 toy 样本随机拆成研究训练/测试集。
超过 16 个样本的批量工作在计算节点执行，继续使用 `scripts/slurm/generate_array.sbatch`。

同一个全局样本索引固定选取排序后 `idx % n_assets` 的资产，随机种子由基础种子与索引确定，
与分片数无关。改变资产清单就改变数据版本。代码、配置、资产内容、视角文件与 harness 构建内容
进入分片指纹；只有指纹一致且分片文件存在时才跳过完成的分片。失败或输入不同必须使用新目录。

`joint_id: "*"` 在生成时展开为当前资产的关节列表。动作按序执行，关节状态持续存在，
不是每个动作都从 rest 开始；当前配置用于采集，不是专家策略。

## 本地资产清单

在生成配置中设置：

```yaml
asset_source: {kind: catalog, manifest: data/assets/catalog.json}
```

清单格式：

```json
{
  "version": 1,
  "source_id": "my-dataset-version",
  "assets": [
    {
      "spec": "specs/object-001.json",
      "split_group": "upstream-original-object-001",
      "origin": "原始发布页及对象 ID",
      "license": "实际适用的资产许可及条款位置"
    }
  ]
}
```

`spec` 相对于清单目录，内容为 `AssetSpec` JSON（可由 `to_dict(spec)` 保存）。
`split_group` 必须把同一原物体的缩放、材质、静态/动态版本等近重复资产归到一起，
不同来源库里重复收录的对象也应使用相同组。`source_id` 应包含上游版本信息。
导入检查 ID、层级循环、材料、尺寸、关节引用、轴和限位；浏览器检查再确认参考程序满足配置约束。

清单支持 box 与嵌入式三角网格 `AssetSpec`。ReplicaCAD 的窄范围 URDF/GLB 转换已实现，
并通过独立运动学审计；命令与边界见 [真实网格 benchmark](mesh-benchmark.md)。

## 标签资格

`corruptions` 记录施加的操作，不等于标签已经合格。`dataset.quality` 另行记录资格：

- 面数扰动：实测面数确实超过配置的预算才成立。
- 部件偏移：参考比较的 rest 状态下，指定部件的几何约束必须违反。
- 关节轴错误：指定关节的已配置档位中，至少一个档位下指定部件的几何约束必须违反；
  真实资产可能上限为零，因此配置必须覆盖下限、中点和上限。
- 多缺陷：同时检查每种扰动单独施加后的结果和最终组合结果，避免把另一个扰动的效应误作该标签成立。
- 正常样本：必须通过全部已配置检查。这个结论只覆盖配置中的约束，不保证所有物理与几何性质。

缺少所需 checker/状态标为 `unsupported`，未生效标为 `invalid`；整个样本进入 `quarantined`。
不合格样本仍保留在原始分片中。执行错误和资格隔离分别计数；生成异常会使分片标为 `incomplete`
并让命令失败，不能静默丢掉样本后宣称整批完成。
原始资产相对自身通过几何比较，不代表其水密性、碰撞、物理合理性已获认证；仍需人工抽查。

## 固定划分、导出和基线

至少收集三个独立资产组后，生成确定性的组划分（约 80/10/10，小数据每个留出集至少一组）：

```bash
python -m turnitover make-splits --shards data/runs/assets-v1/shard-*.tar --seed 42 --out data/splits-v1.json
python -m turnitover prepare-dataset --shards data/runs/assets-v1/shard-*.tar \
  --splits data/splits-v1.json --out data/prepared/assets-v1
python -m turnitover evaluate-dataset --dataset data/prepared/assets-v1 \
  --split test --out output/runtime-assets-v1
```

也可以手工编写明确的组划分，例如单资产工程验收全部放在 `test`：

```json
{"version": 1, "groups": {"toy:toy_cabinet": "test"}}
```

这不产生训练/验证集，不构成泛化实验。自动划分目前仅按资产组，类别留出、未见腐蚀组合划分需要另行设计。

导出目录包括：

| 文件 | 用途 |
| --- | --- |
| `train/validation/test.inputs.jsonl`（实际以点分隔，如 `test.inputs.jsonl`） | 模型可见参考图路径、观测历史、上下文与预算约束 |
| `train/validation/test.gold.jsonl` | 特权缺陷标签、模板诊断、代码、checker 与来源记录；不能作为 judge 输入 |
| `images/` | 参考和观测 PNG，路径相对导出目录 |
| `quarantine.jsonl` | 不合格样本及其原因 |
| `manifest.json` | 数量、类别覆盖、输入分片清单、划分与限制 |

重复样本、缺失图片、缺失组划分、同一资产/参考程序跨集合、未完成分片都会被拒绝。
失败导出留下 `incomplete` 清单供排查，评测器拒绝读取。旧 v1 样本仍可读取，但必须重新生成才能用于新导出。

Gold 模板诊断是未来 oracle 反馈与结构化监督的原料。它不引用当前观测证据，必须经过观测暴露审核
才能作为诊断 SFT 目标；当前不导出伪造的 action 标签，也没有实现 oracle 修复效率上界实验。
可检测性矩阵仍是单动作证据差异代理，关节分支使用特权几何；不能直接把它当作 VLM 可见性标签。

runtime 评测复用现有 `RuntimePolicy`，仅消费导出的公开观测，再与 gold 计算分类型 precision/recall/F1。
`uncertain` 不被排除：对应真缺陷计为漏检，同时报告不确定率。它不测主动动作能力、视觉诊断或修复效率。

## 本轮验收记录（2026-09-12）

使用 `configs/generate_dataset.yaml` 的种子 1234 运行 16 个样本：

| 项目 | 实测 |
| --- | --- |
| 正常 / 单缺陷 / 组合缺陷 | 4 / 6 / 6 |
| 执行成功 / 生成异常 | 16 / 0 |
| 标签合格 / 隔离 | 15 / 1 |
| 隔离原因 | 样本索引 6 增加面数但未超过预算，其组合样本整体隔离 |
| runtime 基线 | 15 个合格样本上 2 个 fail、13 个 uncertain |
| runtime 标签 | 2 TP、0 FP、0 FN |
| 关节轴 / 部件偏移 | 各 7 FN；runtime 策略不具备这些诊断能力 |

这是同一个 toy 柜子的工程验收，不是研究测试集，不报告泛化结论。
正常样本得到 `uncertain` 符合 runtime 基线的边界：预算未超限不等于整个对象已验证正确。

本地输出：`data/runs/dataset-pilot-v2/`、`data/prepared/toy-pilot-v2/`、
`output/runtime-toy-pilot-v2-final/metrics.json`。这些运行文件默认不进 Git。
若需复现整条链路，按上文单资产 `test` 划分执行生成、导出、评测；使用新的输出目录。

本轮测试：82 个单元测试与 9 个浏览器测试通过，包括跨分片稳定性、组合样本、正常样本、
无效标签、输入/真值隔离、跨集合泄漏、目录覆盖拒绝及不确定样本计为漏检。
测试时设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，避免小型几何检查的线程过度竞争。

后续已完成 ReplicaCAD mesh/URDF 转换验收，见 [真实网格 benchmark](mesh-benchmark.md)。
后续也已完成暴露教师、动作/成对诊断导出、Qwen 能力诊断与付费 Sol 修复 pilot。
尚需独立多关节资产扩充、逐轮 oracle 上界，以及固定证据诊断训练与动作行为克隆。
当前状态及证据边界见 [研究状态](research-status.md)；尚未启动模型训练。
