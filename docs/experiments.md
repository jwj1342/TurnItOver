# 基本实验脚手架与可复现示例

当前实现覆盖 RP 中的数据构造、浏览器观测、确定性检查以及预算约束的 verifier 内循环。
尚未完成资产库转换、训练、oracle 反馈上界实验、生成—修复外循环和真实照片留出评测；
下面的合成示例用于检查实现与证据链，不代表论文实验结论。

## 两个完整示例

```bash
source scripts/setup_env.sh
python scripts/run_examples.py --out output/my-examples
# 也可通过集群 CPU 任务执行：
sbatch scripts/slurm/run_cmd.sbatch python scripts/run_examples.py --out output/my-job-examples
```

需要完成 README 中的环境安装与 web 构建，并且系统有 FFmpeg。没有 FFmpeg 时可加
`--no-video` 只输出截图。输出目录必须为空或不存在，避免覆盖历史实验。

脚本固定构造同一个柜子的参考程序和两个独立候选，默认渲染 384×384、12 fps，
转台视频 2 秒、三个关节分别运动的视频合计 6 秒。每个候选执行 12 次预算内观测，
包括运行时统计、两个视角、三个关节的上限驱动、状态变化查询与复位。

| 案例 | 修改 | runtime verifier 预期 | 独立 gold audit 预期 |
| --- | --- | --- | --- |
| triangle-budget | 两个抽屉网格细分到 16×16×16，总面数 6228，阈值 5000 | `fail`，引用实测统计 | 运行时预算不通过 |
| joint-axis | 门的旋转轴由 Y 轴改成 Z 轴 | `uncertain`，运行时预算检查不能判断运动是否正确 | 静止状态一致，门驱动到上限后几何不一致 |

脚本会检查这些关键结果；如果浏览器、审计或预期结果有误，会返回非零退出码，
并在 `summary.json` 中保留错误。`uncertain` 在第二例是基线能力边界，不是正确性通过。
几何审计使用参考程序导出的证据，不向 judge 暴露参考程序源码或腐蚀标签。

每轮输出包括顶层 `index.html`、`summary.json`、参考预览，以及两个案例各自的
`preview/`（截图、视频、源程序、清单）和 `verification/`（轨迹、截图、判定、独立审计）。
仓库保留了 [一次完整输出](../output/research-examples/index.html)；其他输出目录默认忽略。
下载目录后用浏览器打开 HTML 即可查看；GitHub 文件页本身不会执行 HTML。

## 数据与可检测性实验

```bash
python -m turnitover generate --config configs/generate_toy.yaml --n-samples 4 --out data/runs/example-data
python -m turnitover inspect data/runs/example-data/shard-0000.tar --n 4
python -m turnitover detectability --config configs/detectability_toy.yaml --out data/runs/example-detectability.json
```

批量数据任务使用 `scripts/slurm/generate_array.sbatch`。生成配置包含显式种子、腐蚀类别、
观测动作、预算、相机与检查阈值；样本保存程序、标签、轨迹和检查结果。
可检测性矩阵测的是指定阈值下的证据变化，不等同于 VLM 的检测准确率。

## 模型实验

参见 [verifier 使用说明](verifier.md)、[模型配置](models.md) 与
[本地 Qwen 实测记录](verifier-smoke.md)。固定、随机与主动观测使用相同动作空间和预算，
可通过 `--actions` 限制动作类型，通过 `--seed` 控制随机基线。
模型调用次数、token 与耗时单独保存，比较成本时不能只比较观测数量。

本地 Qwen3-VL-2B 三次实测均输出无效判定，均被验证层拒绝。没有执行付费 API 请求。
后续应在固定数据划分与提示词下比较不同模型，报告无效响应率与不确定率，
再增加按缺陷分组的准确性指标；不能删除失败响应后只汇总成功样本。

## 与 RP 的差距

目前可复现的是工程闭环和合成样例。RP 的完整实验还需要：真实资产与照片输入、
按类别划分的数据、多个种子的预算与动作消融、检测与定位指标汇总、
oracle 诊断/观测反馈对生成器的上界测试，以及修复循环效率测量。
当前 HTML 报告和 JSON 轨迹为这些实验提供输入与审计依据，不代替上述评测。
