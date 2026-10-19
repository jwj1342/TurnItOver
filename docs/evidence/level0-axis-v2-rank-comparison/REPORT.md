# PartNet-Mobility Level 0 v2：Qwen3-VL-2B LoRA rank 对照

2026-10-18 实际完成两组独立训练和 Train/Held-out 全量评测。下一轮小范围实验优先选择 **rank=4、alpha=8**：它在本次固定留出集得到 11/12，rank=8 为 10/12，可训练参数减半。rank=4 已足以拟合当前训练任务，并在本次留出集上优于底座；尚未完全解决留出泛化。没有观察到 rank=8 的泛化收益。单 seed、6 个留出物体下，一例差异不足以证明 rank=4 稳定优于 rank=8。

## 数据与固定划分

- 来源：`https://github.com/jwj1342/TurnItOver`，`laq/sft-dataset` 提交 `d5cbc085a4dbb7b573aa17f42ec67b322d0003b3` 的 `docs/evidence/partnet-mobility-level0-axis-v2/`。执行 `git ls-remote` 核实远端分支 SHA；仅用 `git archive` 导出该目录，没有合并 dataset 代码，也未修改 dataset 工作树。
- 413 个导出文件逐一与指定提交 Git blob 身份匹配；其中 manifest 列出的 412 个文件 SHA 全部匹配。审核、训练清单、Gold、私有 train targets 的 56 个 case 完全一致且均合格；每个原始物体各有 clean/pass 与 joint_axis/fail。来源 model ID、archive SHA、几何指纹、split group 均无重复，全部属于上游 Train，未使用 frozen Test。
- 复用 `train_manifest.jsonl` 的成员与 `input_preview`，`object_splits.json` 的原始物体分组；使用原有 542×542 四宫格。监督标签取 `human_review.jsonl`，私有 Gold/targets 仅供离线核对。审核描述包含代理根据用户逐图确认代录的文字，未将其当作额外训练监督。
- 22 个训练物体/44 例，6 个留出物体/12 例，clean/axis 各半，配对不跨集合。先排除旧 binary SFT/fixed-evidence 五物体、继承审核物体及 source catalog 中有 `previous_motion_review` 的物体，再按 `SHA256(20261017:split_group)` 排序取六个新物体。该“新增”针对项目记录，无法证明底座预训练从未见过对应资产。
- 留出：`PM_22433, PM_41003, PM_45213, PM_45305, PM_45621, PM_46130`。完整成员见 [split.json](split.json)，SHA-256 为 `5881c1d371827b6920ba4159a9bdf707ddc7ac9a33bb3d0322378b5968336a54`。21:57:28（Asia/Shanghai）已冻结，早于训练完成和全部评测；两组始终使用同一文件，未按结果换划分或调参。
- 审核/SHA/来源明细见 [data-audit.json](data-audit.json) 和 [source-identity.json](source-identity.json)。上游 manifest 中包外绝对路径只保留其声明哈希，本次未重新读取这些外部文件，也未重跑图像渲染。

## 结果

分母为样本数；同物体两例相关，独立留出单位实际仅 6 个。

| 方法 | Train 总体 | Train clean | Train axis | Held-out 总体 | Held-out clean | Held-out axis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 未微调 Qwen3-VL-2B | 22/44 (50%) | 22/22 (100%) | 0/22 (0%) | 6/12 (50%) | 6/6 (100%) | 0/6 (0%) |
| LoRA rank=4, alpha=8 | 44/44 (100%) | 22/22 (100%) | 22/22 (100%) | **11/12 (91.67%)** | 6/6 (100%) | **5/6 (83.33%)** |
| LoRA rank=8, alpha=16 | 44/44 (100%) | 22/22 (100%) | 22/22 (100%) | 10/12 (83.33%) | 6/6 (100%) | 4/6 (66.67%) |
| 既有像素时序差分基线 | 44/44 (100%) | 22/22 (100%) | 22/22 (100%) | **12/12 (100%)** | 6/6 (100%) | 6/6 (100%) |

所有模型均无 invalid 输出。底座对全部 56 例均输出 pass；SFT 的收益均来自恢复 axis 检出且保留 clean 正确率。rank=4/rank=8 的留出总体准确率比底座分别高 41.67/33.33 个百分点，比像素基线低 8.33/16.67 个百分点。rank=8 相对 rank=4 低 8.33 个百分点（仅多错一例；配对 exact McNemar 双侧 p=1.0，不支持稳定排序）。像素阈值只由 44 个训练样本的 `(max pass + min fail)/2` 确定，为 `1.3589630126953125`，未读取留出标签拟合阈值。

错误案例（均为 axis 被判为 pass）：

| case | 物体 | rank=4 | rank=8 | 底座 | 像素 |
| --- | --- | --- | --- | --- | --- |
| `l0-29cdda903051` | PM_46130 | 错 | 错 | 错 | 对 |
| `l0-24df3ba7b2a9` | PM_45305 | 对 | 错 | 错 | 对 |

前者四宫格中参考门与候选门展开的位置/方向不同，两组均漏检；后者参考底板与候选底板展开平面不同，rank=8 漏检。其余四个留出 axis，底座也全部漏检，两组 SFT 均答对；所有 Train 错误仅来自底座的 22 个 axis。完整错误 ID、逐例原文及重算分项见 [summary.json](summary.json) 与各方法的 `predictions.jsonl`。

两组 Train 达 100%，但留出仍错 1/2 例，训练与留出差距为 8.33/16.67 个百分点，存在过拟合迹象。rank=4 的逐 epoch 平均在线训练 loss 为 `0.221668 → 0.00001834 → 0.000000193`；rank=8 为 `0.366708 → 0.137088 → 0.121518`，第三 epoch 仍有单步 loss 4.833 的波动，尽管最终 Train 全对。在线 loss 在不同更新状态下计算，不能等同于最终 checkpoint 的统一数据集 loss。未在留出集选 checkpoint，均使用第 3 epoch 最终 adapter。

这些结果支持当前固定视角 Level 0 二分类的有限物体泛化，不能证明模型学会轴语义或主动取证；像素基线更好也提示任务存在足够强的像素差异信号。下一轮优先用 rank=4 保持其余协议固定，小范围补充 seed 复核稳定性；不据此更换本次留出划分。

## 配置与成本

两组均从同一个原始 `Qwen3-VL-2B-Instruct` 独立加载，没有加载历史 adapter。模型权重 SHA-256 为 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`；所有底座文件哈希见 [model-sha256.txt](model-sha256.txt)。

共同配置：seed `20261017`，LR `2e-4`，3 epochs/132 optimizer steps，batch=1，无梯度累积；AdamW β=(0.9,0.999)、ε=1e-8、weight_decay=0.01，固定 LR、clip_grad_norm=1。LoRA target=`q_proj,k_proj,v_proj,o_proj`，dropout=0、bias=none、CAUSAL_LM，alpha/r=2。BF16、SDPA，assistant-only loss（pass/fail 及模板结束 token）。唯一设计变量为 rank/alpha。

运行模式限制：沿用的 trainer 调用了 `gradient_checkpointing_enable()`，但没有显式调用 `model.train()`；在实际 transformers 5.5.0 / peft 0.18.0 环境核查，底层 Qwen 保持 eval mode，所以 checkpoint 重计算未生效。本模型 attention dropout=0，且没有非零 Dropout 模块；eval mode 不禁用 autograd，两组仍实际执行了反向传播和参数更新。两组模式完全一致，下面的成本数据对应这一实际模式；本次保留已完成的运行，不在观察留出结果后改变训练行为。

模型只接收一张四宫格和以下固定文本，不接收 case/asset ID、源清单 prompt、审核运动描述、Gold 或几何参数：

> The image is a four-panel comparison. Top row: normal reference at rest, then limit. Bottom row: candidate at rest, then limit. Does the candidate motion match the reference? Answer exactly one word: pass or fail.

图像顺序固定为 reference rest/limit、candidate rest/limit，沿用同一底座 processor，`image_grid_thw=[[1,34,34]]`、`pixel_values=[1156,1536]`；评测前缀 345 tokens，greedy generation、max_new_tokens=4，同一严格单词判分规则。

在 `laq@10.201.129.199` 两张空闲 RTX 5090 D v2 上并行执行，rank4 使用 GPU0，rank8 使用 GPU1；同一已验证 venv（Python 3.11.15、torch 2.10.0+cu128、transformers 5.5.0、peft 0.18.0）。未安装/升级环境；seeded CUDA witness 成功。环境见 [env-spec.json](env-spec.json) 和 [pip-freeze.txt](pip-freeze.txt)。

| 配置 | 可训练参数 | 训练墙钟（含加载、保存） | 峰值 torch allocated | adapter 文件 | Train+Held-out 评测墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank=4 | 1,605,632 | 16.25 s | 5.700 GiB | 6,455,336 B | 7.08 s |
| rank=8 | 3,211,264 | 16.08 s | 5.719 GiB | 12,878,000 B | 7.71 s |

按占用 GPU 的进程墙钟估算，训练合计约 32.33 GPU-seconds（0.009 GPU-hours），含两组和底座评测合计约 53.05 GPU-seconds（0.0147 GPU-hours）；这不是 GPU 内核实际忙碌时间。不含 SSH、复制、SHA 审计和测试。均为现有自有服务器，无付费模型调用。速度差异很小，不能声称 rank4 更快；确定的资源差异是参数与 adapter 大小约减半。每组完整 loss、run.json、checkpoint 和日志保留于本地 `output/level0-axis-v2-rank-comparison/`，远端保留于 `/home/laq/experiments/level0-axis-v2-rank-comparison/output/`。Git 仅提交小型日志/预测/配置/报告，不提交权重。

## 复现与修改范围

从本训练仓库根目录导出指定提交（若本地没有该对象，先 fetch `origin laq/sft-dataset`；无需 merge）：

```bash
SOURCE=d5cbc085a4dbb7b573aa17f42ec67b322d0003b3
mkdir -p data/level0-axis-v2-source
git archive "$SOURCE" docs/evidence/partnet-mobility-level0-axis-v2 | tar -x -C data/level0-axis-v2-source
PYTHONPATH=. ../TurnItOver/.venv-qwen/bin/python scripts/prepare_level0_axis_sft.py \
  --dataset data/level0-axis-v2-source/docs/evidence/partnet-mobility-level0-axis-v2 \
  --out output/level0-axis-v2-reproduce-data --seed 20261017
cmp output/level0-axis-v2-reproduce-data/split.json docs/evidence/level0-axis-v2-rank-comparison/split.json
```

原始运行：通过 rsync 将本次 `scripts/`、`turnitover/`、完整源数据和冻结 split 分别放至远端 ROOT 的 `code/`、`data/source/`、`data/frozen/split.json`。然后在服务器运行以下命令；输出路径必须全新：

```bash
ROOT=/home/laq/experiments/level0-axis-v2-rank-comparison
cd "$ROOT/code"
DATASET="$ROOT/data/source" SPLIT="$ROOT/data/frozen/split.json" \
MODEL=/home/laq/experiments/kimi-v4-increment-v1/models/Qwen3-VL-2B-Instruct \
PYTHON=/home/laq/experiments/kimi-v4-increment-v1/venv/bin/python \
OUT="$ROOT/output" bash scripts/run_level0_axis_rank_comparison.sh
```

runner 固定 seed/epoch/LR/rank/alpha，调用现有训练、评测、像素脚本。将远端 output rsync 回本地后，可从保留的轻量证据重算全部指标并核对训练协议：

```bash
PYTHONPATH=. ../TurnItOver/.venv-local/bin/python scripts/summarize_level0_axis_rank_comparison.py \
  --dataset data/level0-axis-v2-source/docs/evidence/partnet-mobility-level0-axis-v2 \
  --split docs/evidence/level0-axis-v2-rank-comparison/split.json \
  --root docs/evidence/level0-axis-v2-rank-comparison
PYTHONPATH=. ../TurnItOver/.venv-local/bin/python -m pytest -o addopts= tests/unit -q
```

验证：**252 passed, 2 skipped**，详见 [tests.txt](tests.txt)。新增 8 个回归检查覆盖确定性配对划分、历史物体排除、图像篡改、标签不一致、重复源身份、不合格物体、错误配对、清单过滤与旧版兼容。shell 语法检查通过，summary 对全部逐例预测、标签、split hash、132 步顺序和两组共同配置核对通过。

代码修改仅限：`scripts/prepare_level0_axis_sft.py`、`turnitover/oracle/level0_axis.py`、新增 `turnitover/oracle/level0_axis_v2.py`（审计/划分）、新增 `tests/unit/test_level0_axis_v2.py`；现有 train/evaluate 脚本仅补充配置/成本日志；新增 `scripts/run_level0_axis_rank_comparison.sh` 和 `scripts/summarize_level0_axis_rank_comparison.py`。其余提交文件均为本实验轻量证据。未改变训练框架、未引入其他任务。

独立新上下文审计于 2026-10-19 完成，结论为 WARN：数值与数据完整性通过，保留单 seed/小留出集及实际训练模式限定；见 [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md) 和 [runtime-mode-audit.json](runtime-mode-audit.json)。审计为 same-family/provisional。
