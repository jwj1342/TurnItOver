# 运行环境

沿用 `.aris/compute/ssh-kimi-v4.md` 指向的既有环境，不安装或升级软件。
远端 `laq@10.201.129.199`，Python `/home/laq/experiments/kimi-v4-increment-v1/venv/bin/python`，torch 2.10.0+cu128、transformers 5.5.0、peft 0.18.0。
基座 `/home/laq/experiments/kimi-v4-increment-v1/models/Qwen3-VL-2B-Instruct`，训练前逐文件SHA记录在run.json。
独立目录 `/home/laq/experiments/partnet-action-sft-v1/`，GPU0（启动前67MiB，GPU1为24MiB），四线程CPU，完全离线，不调用付费API。

预检 seeded CUDA witness：seed=20261017，8×8随机矩阵 `sum(x@x)=16.453990936279297`，成功执行真实CUDA kernel。
训练与四策略推理使用同一模型环境；每次冻结最终诊断时禁用Action adapter。Chromium留在本机已有 `.venv-local` 和固定浏览器缓存，模型请求经localhost SSH隧道传送prompt与图像。

本机测试使用 `PYTHONPATH=.:../TurnItOver/.venv-local/lib/python3.12/site-packages ../TurnItOver/.venv-qwen/bin/python -m pytest -o addopts= tests/unit -q`；附加site-packages仅供测试，不改变远端训练环境。

首次模型桥接预检发现入口导入渲染器而间接依赖 scipy；已将模型适配独立为 `turnitover/models/action_qwen.py`，远端不用安装模拟器依赖。首轮基线 transport 受本机 HTTP 代理影响返回 502，立即停止；`output/partnet-action-sft-v1/baselines/` 留档且排除正式成绩。正式桥接强制 loopback 并显式绕过代理，保持请求只到 SSH 本地转发。工程 fake-diagnoser smoke 只验证真实浏览器重放，不计入模型实验。

全部实验结束后，已停止本轮两项模型服务与 SSH 转发；远端GPU恢复为0号67MiB、1号24MiB，未停止他人进程。数据、checkpoint和日志继续保留。
