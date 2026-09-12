# 计算集群配置

基础代码不依赖某个机构或集群。以下区分通用 Slurm 环境和仓库已有的受管环境配置；无集群的合作者使用 [个人电脑配置](local-setup.md)。

## 通用 Slurm 环境

在共享文件系统中克隆仓库，并按所在集群规则准备 Python 3.12、Node.js、FFmpeg 与浏览器所需系统库。若使用环境模块，将本站模块配置集中写在 `scripts/setup_env.sh`，不要分散到各作业脚本或 shell 全局配置。

允许访问公共软件源时，在允许安装软件的联网节点执行：

```bash
bash scripts/bootstrap_local.sh
source scripts/setup_local.sh
```

受限软件源、无网络环境或计算节点与安装节点不兼容时，应按管理员要求准备依赖与浏览器；不要直接使用不符合本站规则的下载命令。`.venv-local/`、`.cache/ms-playwright/`、`web/node_modules/` 与 `web/dist/` 须在计算节点可访问。使用环境模块提供的 Python/Node 时，作业也需要加载同样的环境；可在通用作业脚本中激活本地环境之前调用配置好的 `scripts/setup_env.sh`。

从仓库根目录提交：

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  scripts/slurm/run_portable.sbatch python scripts/check_browser.py
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  scripts/slurm/run_portable.sbatch python scripts/run_examples.py --out output/cluster-examples
```

账户与分区需替换；不要求这些选项的集群可以省略。`run_portable.sbatch` 不含机构账户或 GPU 假设，使用已准备好的共享环境，不在计算节点下载依赖。基础示例使用 CPU 渲染；可通过 `--mem`、`--time` 和 `--cpus-per-task` 调整资源。日志输出到 `logs/`。

## 已有受管环境配置

`scripts/setup_env.sh` 和 `scripts/bootstrap_login.sh` 保留了项目已有的 Environment Modules 与 Alliance wheelhouse 配置，**不是通用安装入口**。在匹配的环境中：

```bash
bash scripts/bootstrap_login.sh
source scripts/setup_env.sh
sbatch --account=YOUR_ACCOUNT scripts/slurm/smoke_browser.sbatch
CONFIG=configs/generate_toy.yaml sbatch --account=YOUR_ACCOUNT scripts/slurm/generate_array.sbatch
```

此路径从 wheelhouse 安装依赖（`pip install --no-index`），不使用 PyPI；新增依赖先用 `avail_wheels` 核验。登录环境位于 `.venv/`，计算作业在 `$SLURM_TMPDIR/venv` 创建环境。旧的 `.sbatch` 文件含已有账户默认值，必须通过命令参数覆盖或按本站修改；本地 Qwen 作业还包含具体 GPU 资源名，需额外匹配本站硬件。

## 网络、存储和复现

- 网络 API 只在允许联网的节点执行。计算节点是否联网由本站决定。
- 浏览器、模型和前端依赖尽量在作业开始前准备好；模型权重放在 `models/`。
- 每次实验使用独立输出目录，不让多个任务写入同一路径。
- 代码与精选示例通过 Git 同步；`.env`、虚拟环境、大型数据和权重不提交。
- 共享或临时存储的清理策略、配额与备份规则以本站为准。
- `squeue` 查看任务，`sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS` 查看资源与退出情况。

数据引擎按分片和样本索引生成稳定随机种子；已完成的分片可跳过。verifier 单次运行不支持断点恢复，失败后使用新输出目录重新运行。
