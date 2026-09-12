#!/bin/bash
# 个人电脑或通用 Linux 环境安装入口，需要联网；不加载集群模块。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_local.sh
TIO_PYTHON="${TIO_PYTHON:-python3.12}"
"$TIO_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "请使用 Python 3.12"'
command -v npm >/dev/null || { echo "请先安装 Node.js 和 npm" >&2; exit 1; }
if [ ! -f "$TIO_LOCAL_VENV/bin/activate" ]; then
  "$TIO_PYTHON" -m venv "$TIO_LOCAL_VENV"
fi
source "$TIO_LOCAL_VENV/bin/activate"
python -m pip install -r requirements.txt
python -m playwright install chromium-headless-shell
(cd web && npm ci && npm run build)
echo "安装完成。执行 source scripts/setup_local.sh，然后 python scripts/check_browser.py"
