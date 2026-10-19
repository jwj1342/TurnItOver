# Action-only SFT：入口完成，原始数据缺失，实验未执行

后续用户授权改用既有 PartNet，实际训练和闭环已完成，见 [PartNet 实验报告](../partnet-action-sft-v1/REPORT.md)。本文件保留最初 ReplicaCAD 缺包预检的历史状态。

2026-10-19；分支 `laq/sft-training`；开始 HEAD `b752847`，工作区干净，领先 origin 一个本地提交。保留该提交及所有历史实验；本次仅增加 Action-only 入口、测试和审计记录，不推送、不提交权重。

**未达到实际实验完成标准。** 本机三个工作树与可访问的 `laq@10.201.129.199` 未找到指定原始 bundle。快照明确不包含图片、训练数据和完整渲染程序；历史 `/scratch/jwj/TurnItOver` 在该服务器不存在。两张 RTX 5090 D v2 可用（检查时显存 67/24 MiB），缺口是数据而非计算资源。原数据路径已向用户询问。

实际预检以指定快照目录运行 trainer，在读取 `train.actions.jsonl` 时报 FileNotFoundError，见 [preflight.txt](preflight.txt)。没有执行 GPU 优化、保存或加载 adapter、Chromium 评测，也没有本轮 loss 或能力结果。不存在 checkpoint 的情况下不运行伪闭环，不将其他数据或重新生成的教师当作原监督。

数据来源、物体隔离、可见前缀冲突及预算审计边界见 [AUDIT.md](AUDIT.md)、[data-audit.json](data-audit.json)。现有 Validation 只有一个独立物体、16 个案例。历史 manifest 中全部 split 汇总的暴露曲线不作本轮 Validation 结果。

代码复用 `supervision.load_action_groups`、`training.tokenize_example/weighted_assistant_loss`。训练从原底座独立初始化 rank=4/alpha=8、LR=2e-4、seed=20261017、3 epochs；每个组的全部动作概率加权反向累积，然后一次更新，与整组打包的目标梯度等价。没有诊断/停止目标，没有任取唯一 Gold。组内分支顺序执行，dropout=0。

闭环复用 `run_verification`、`execute_observation`、原 witness Fixed/Random planner 和既有真实历史反事实评分。四策略共用禁用 adapter 的冻结底座最终诊断器；全体策略推理结束后才执行私有评分。逐次动作、图像、prompt、原始响应和 episode 将写入独立输出目录。此处描述代码路径，不代表已运行验证。

恢复既存数据后（不要使用快照目录代替完整数据）：

```bash
# 在已验证的 qwen 环境；路径替换为既存完整原包，不重新生成
PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/train_qwen_action_sft.py \
  --dataset /existing/oracle-action-distributions-v1 \
  --model /existing/Qwen3-VL-2B-Instruct --out output/action-only-sft-v1/train \
  --epochs 3 --seed 20261017 --learning-rate 2e-4
PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_action_sft_closed_loop.py \
  --benchmark /existing/replicacad-v1 --oracle /existing/oracle-v1 \
  --asset-workspace /existing/asset-workspace \
  --harness-workspace /existing/built-workspace \
  --model /existing/Qwen3-VL-2B-Instruct \
  --adapter output/action-only-sft-v1/train/checkpoint \
  --out output/action-only-sft-v1/closed-loop --budgets 2 4 6
```

以上是待执行命令，尚未验证完整端到端环境；需要先验证原数据行、图像 SHA、成员与 split 身份以及 Validation 预算区分度，然后一次训练，不按 Validation 收益调参。恢复原包后还需以最终 checkpoint 在固定 Train/Validation 公共前缀上测组内加权 NLL/预测动作概率质量，与底座比较；训练 loss 本身不能代替最终 checkpoint 的可学习性检验。

两个研究问题目前都为 **未验证**。不能将数据不可访问归因为监督不可识别、训练未学会或无需主动选择。这三类解释须等真正的固定条件实验后分别检验。

验证命令（本机已执行）：

```bash
PYTHONPATH=.:../TurnItOver/.venv-local/lib/python3.12/site-packages \
  ../TurnItOver/.venv-qwen/bin/python -m pytest -o addopts= tests/unit -q
```

qwen 环境缺少 pytest，因此仅对测试进程附加现有 browser venv 的 site-packages，没有安装或改变训练环境。完整结果见 [tests.txt](tests.txt)。组内加权梯度、原始导出与在线前缀一致、未来观察排除、动作阶段拒绝提前诊断、冻结诊断调用、非法动作计数与截断响应拒绝均有检查。真实 GPU/浏览器与 checkpoint 加载尚未验证。
