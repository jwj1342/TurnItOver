# 个人电脑配置

个人电脑可以运行合成数据小样本、浏览器观测、截图与视频、离线验证，以及 OpenRouter 等远端模型调用。只有本地 Qwen 推理需要当前适配器支持的 CUDA GPU；Mac 或无独立 GPU 的电脑可以使用远端 API。

## 前置软件

- Python 3.12，包含 `venv` 与 pip。
- Node.js 22 或 24，包含 npm。
- FFmpeg，并包含 `libx264` 编码器；只输出截图时不需要。
- Git 和 Bash。Windows 使用 WSL2，在 WSL 的 Linux 文件系统内克隆仓库和安装软件，避免混用 Windows 与 Linux 的 Python、Node 或浏览器。

Linux 和 macOS 均提供安装路径；本次实际验证环境为 Linux，沿用宿主提供的 Python/Node 和 pip 软件源，尚未在 macOS、WSL 或完全独立的公共包源环境中实机验证。操作系统还需满足 [Playwright 的支持要求](https://playwright.dev/python/docs/intro#system-requirements)。

## 安装

从仓库根目录执行：

```bash
bash scripts/bootstrap_local.sh
source scripts/setup_local.sh
```

若 Python 3.12 的可执行文件名不是 `python3.12`：

```bash
TIO_PYTHON=/path/to/python3.12 bash scripts/bootstrap_local.sh
```

脚本创建 `.venv-local/`，使用 pip 配置的包索引安装 `requirements.txt`，下载 Chromium 到 `.cache/ms-playwright/`，再执行 `npm ci` 与网页构建。默认公共环境使用 PyPI；有镜像或私有包源时遵循自己的 pip 配置。固定依赖与集群共用，避免维护两套版本。

Ubuntu/Debian 若提示缺少浏览器共享库，可在激活环境后执行：

```bash
python -m playwright install-deps chromium
```

该命令安装系统软件，可能需要管理员权限；不在 bootstrap 中自动执行。FFmpeg 请通过操作系统包管理器安装，再用 `ffmpeg -version` 检查。浏览器安装机制见 [Playwright 文档](https://playwright.dev/python/docs/browsers)。

## 检查与运行

```bash
python scripts/check_browser.py
pytest
pytest -m browser
python scripts/run_examples.py --out output/my-examples
# 不生成视频时：
python -m turnitover preview --no-video --out output/my-screenshots
```

新终端中先回到仓库根目录，再执行 `source scripts/setup_local.sh`。所有 `python -m turnitover ...` 命令与集群路径共用，无需安装集群模块或 Slurm。

复制 `.env.example` 为 `.env` 后配置模型服务商、模型 ID 与密钥；离线示例不需要 API 密钥。详见 [模型配置](models.md)。本地 Qwen 的现有依赖文件与提交脚本按受管 Linux GPU 环境验证，其他 CUDA 环境需匹配本机 PyTorch 安装，不能直接将集群 GPU 资源名用于个人电脑。
