# TurnItOver

面向大模型生成的三维程序，研究**验证即主动感知**：评判器通过选择视角、驱动关节和查询运行时信息，在有限观测预算下获取证据、定位缺陷。研究计划见 [RP.md](RP.md)。

当前已实现合成数据引擎、无头浏览器观测、确定性检查器、截图与视频导出、按角色配置的模型 API，以及支持主动、固定、随机观测的 verifier（验证器）。另有单次照片到程序的生成入口和离线运行时检查基线。已完成真实资产小型 benchmark、自动监督导出及 GPT-5.6 Sol 四条件修复试验。评判器训练、完整 RP oracle 上界与真实照片评测尚未完成。

## 当前研究进展

合作者请先读 [研究状态与接手指南](docs/research-status.md)：包含研究问题、已验证证据、结论边界、下一轮完成标准和从零复现顺序。
[已提交测量快照](results/2026-09-12/README.md) 可直接核对指标、真实模型补丁与费用，无需 API 或 GPU。
修复试验中固定观察成功 1/3，三种额外反馈条件各成功 3/3；仅三个损坏案例，不能作为训练收益或泛化结论。

## 项目结构

```text
turnitover/   Python 包：数据契约、资产、腐蚀、渲染、观测策略、检查器、存储、模型和验证器
web/         Three.js 运行时与 window.harness，构建产物位于 web/dist
configs/     视角、数据生成和可检测性实验配置
scripts/     环境初始化、模型下载、示例入口和 Slurm 作业脚本
tests/       单元测试与需要浏览器的集成测试
docs/        架构、协议、模型配置、实验说明和缺陷分类文档
results/     进入版本控制的轻量测量快照，不含大规模数据或凭证
models/      本地模型权重，仅说明文件进入版本控制
output/      截图、视频、验证报告；保留一组完整示例，其余运行结果默认忽略
```

## 环境配置

两种环境运行同一套代码和实验；没有集群也可以运行截图、视频、离线 verifier 和远端模型 API。基础示例不需要独立 GPU，本地 Qwen 推理另需 CUDA GPU。

### 方法一：个人电脑

准备 Python 3.12、Node.js 22 或 24（含 npm）与 FFmpeg。Linux、macOS 使用 Bash；Windows 请使用 WSL2，在 WSL 内安装依赖并克隆仓库。

```bash
bash scripts/bootstrap_local.sh
source scripts/setup_local.sh
python scripts/check_browser.py
python scripts/run_examples.py --out output/my-examples
```

安装脚本在 `.venv-local/` 创建 Python 环境，从包索引安装固定依赖，下载 Chromium 并构建网页运行时。Linux 如果缺少浏览器系统库，按 [个人电脑配置](docs/local-setup.md) 安装。新终端重新执行 `source scripts/setup_local.sh` 即可。

### 方法二：计算集群

先按 [计算集群配置](docs/cluster.md) 准备共享环境，再通过调度器提交实验。Slurm 示例：

```bash
# 将 YOUR_ACCOUNT、YOUR_PARTITION 替换为自己的资源配置；不需要的选项可以省略。
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  scripts/slurm/run_portable.sbatch python scripts/run_examples.py --out output/cluster-examples
```

通用作业脚本不指定机构、账户或模块版本。仓库另保留已有受管环境的脚本，使用前需按本站配置调整；计算节点是否联网、允许哪些软件源，以所在集群规则为准。

环境就绪后，可执行数据与可检测性实验：

```bash
python -m turnitover generate --config configs/generate_toy.yaml --shard 0/2 --n-samples 8
python -m turnitover detectability --config configs/detectability_toy.yaml --out data/runs/detect_toy.json
python -m turnitover inspect data/runs/toy_smoke/shard-0000.tar --n 4
python -m turnitover render-docs
```

以下命令均假设已激活对应环境，并从仓库根目录执行。集群上的批量任务请放入计算作业。

## 截图与视频

使用已构建的网页运行时和 Chromium，将实际渲染结果导出到 `output/`：

```bash
python -m turnitover preview
# 或在计算节点执行：
sbatch scripts/slurm/run_portable.sbatch python -m turnitover preview
```

默认创建带时间戳的目录，包含：

- `index.html`：离线展示页面，下载整个目录后用浏览器打开。
- `screenshots/`：12 个静止视角，以及每个关节运动到上限的截图。
- `overview.png`：带标签的截图总览。
- `turntable.mp4`：360 度转台视频，默认 4 秒。
- `joints.mp4`：各关节分别运动的视频，默认每个关节 4 秒。
- `program.ts`、`manifest.json`：源程序、相机、取景范围、关节限位、运行时统计和运行元数据。

视频由系统 FFmpeg 的 `libx264` 编码，使用浏览器兼容的 H.264/yuv420p 格式。按固定帧步长采样，播放速度不受渲染耗时影响。可用 `--ffmpeg /path/to/ffmpeg` 指定编码器，或用 `--no-video` 只导出 PNG。

常用参数：`--out output/my-preview --size 640 --fps 24 --seconds 4`。输出目录必须为空或不存在。除仓库保留的完整示例外，生成媒体默认不进入版本控制。使用 `--program path/to/object.ts` 可预览其他符合程序 ABI 的候选。连续转台与运动仅用于展示，研究策略仍使用离散动作空间。

## 模型 API 与照片输入

参照 [.env.example](.env.example) 配置本地 `.env`，分别为生成器（generator）、评判器（judge）和诊断角色（diagnosis）选择服务商与完整模型 ID。支持 OpenRouter 等服务；现有环境变量优先于 `.env`，密钥不进入版本控制。

```bash
python -m turnitover models-check  # 仅检查本地配置，不调用 API
python -m turnitover reconstruct --image reference.jpg --out output/photo-run
python -m turnitover preview --program output/photo-run/program.ts --out output/photo-preview
```

使用 `reconstruct ... --dry-run` 可在不提供密钥、不请求网络的情况下准备输入。支持的接口、配置覆盖方式和限制见 [模型配置](docs/models.md)；已有重建方案和接入缺口见 [流水线比较](docs/pipeline-options.md)。

## 验证器

仓库自带示例程序，可以直接执行离线检查：

```bash
python -m turnitover verify --program output/research-examples/triangle-budget/preview/program.ts \
  --policy runtime --budget 5 --out output/verify-runtime
# 主动视觉语言模型使用 .env 中的 judge 角色：
python -m turnitover verify --program candidate.ts --reference-image reference.jpg \
  --policy active --budget 8 --out output/verify-active
```

输出包含 HTML 证据页面、结构化判定与修复建议、观测轨迹和模型调用记录。预算耗尽、判定不确定与执行错误分别记录；命令成功结束不等于候选通过验证。

本地 Qwen3-VL 权重放在 `models/`，通过 GPU 作业执行推理。GPU 环境、Slurm 命令及证据边界见 [验证器说明](docs/verifier.md)。目前 Qwen3-VL-2B 的实测判定仍不可靠，详见 [实测记录](docs/verifier-smoke.md)。

## 可复现示例

数据基础设施新增多资产清单、正常/组合腐蚀、标签资格检查、按资产组划分和输入/真值隔离导出，
使用方法见 [数据流水线](docs/dataset-pipeline.md)。已调研关节资产库，并提供 ReplicaCAD 公开关节子集
的固定版本下载脚本，见 [资产调研](docs/asset-sources.md)。已完成六类 ReplicaCAD 网格转换和 102 状态
运动学审计，命令与验收记录见 [真实网格 benchmark](docs/mesh-benchmark.md)。
已导出预算内 oracle 动作监督与成对参考诊断，见 [Oracle 与训练导出](docs/oracle-supervision.md)。VLM 训练尚未实现。

```bash
python scripts/run_examples.py --out output/my-examples
```

该命令生成面数超限、关节轴错误两个合成案例，包含参考与候选的截图、视频、验证轨迹和独立审计。查看 [实验说明](docs/experiments.md) 与 [已保存的示例总览](output/research-examples/index.html)。这些示例用于验证工程链路，尚不构成研究基准。

## 工程原则

项目按以下原则维护；当前实现的证据与局限见 [工程原则检查](docs/engineering-principles.md)。

| 原则 | 实践要求 |
| --- | --- |
| DRY：避免重复 | 共用动作执行、序列化和模型接口，避免多套逻辑产生行为差异。 |
| 关注点分离 | 数据构造、观测策略、浏览器执行、检查与展示分别组织。 |
| SRP：单一职责 | 模块围绕一个明确职责变化，编排入口负责连接组件。 |
| 清晰抽象与契约 | 使用小接口和显式数据契约，固定协议的修改必须同步版本、文档与测试。 |
| 低耦合、高内聚 | 检查器只消费导出证据；策略通过会话接口与浏览器交互。 |
| 可扩展与无状态 | 数据分片使用稳定种子；运行状态限定在单个任务内，任务之间隔离。 |
| 可观测与可测试 | 保存轨迹、判定、错误、模型用量和耗时，并提供单元与浏览器测试。 |
| KISS：保持简单 | 优先使用直接的函数、配置文件和现有集群工具。 |
| YAGNI：避免过度设计 | 只实现当前实验需要的能力，后续研究模块按实际需求增加。 |

## 测试

```bash
pytest             # 单元测试，不启动浏览器
pytest -m browser  # 需要 web/dist 和 Playwright 浏览器
```

## 进一步阅读

- [架构说明](docs/architecture.md)：模块边界与固定契约。
- [协议说明](docs/protocol.md)：浏览器运行时接口与验证记录约定。
- [实验说明](docs/experiments.md)：示例复现与当前研究范围。
- [仓库工作约定](CLAUDE.md)：环境、依赖与提交前检查。
