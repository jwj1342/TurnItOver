#!/bin/bash
#SBATCH --job-name=tio-portable
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# 从仓库根目录提交；按本站要求传入 --account 和 --partition。
# 共享目录中的环境与浏览器须提前安装，计算节点不执行下载。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?请通过 sbatch 提交}"
source scripts/setup_local.sh
if [ ! -x "$TIO_LOCAL_VENV/bin/python" ]; then
  echo "请先按 docs/cluster.md 准备共享环境" >&2
  exit 1
fi
exec "$@"
