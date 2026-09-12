# 在仓库根目录执行 source scripts/setup_local.sh；不加载集群模块。
export TIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TIO_LOCAL_VENV="${TIO_LOCAL_VENV:-$TIO_ROOT/.venv-local}"
export PLAYWRIGHT_BROWSERS_PATH="$TIO_ROOT/.cache/ms-playwright"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$TIO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TIO_DATA_DIR="${TIO_DATA_DIR:-$TIO_ROOT/data}"
if [ -f "$TIO_LOCAL_VENV/bin/activate" ]; then
  source "$TIO_LOCAL_VENV/bin/activate"
fi
