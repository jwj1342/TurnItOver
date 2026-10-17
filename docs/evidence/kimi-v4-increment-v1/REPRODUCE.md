# 复现

在 `laq/sft-training` 工作树运行。来源远程 commit 已固定；下述命令不会写原始数据或冻结 Test。模型及原始 v4 export 是已有本地产物，必须先与本目录 `model-manifest.json`、`data-audit.json` 的 SHA 一致。

```bash
git -C ../TurnItOver-dataset fetch origin laq/sft-dataset
../TurnItOver/.venv-qwen/bin/python scripts/prepare_kimi_v4_increment.py \
  --repo ../TurnItOver-dataset --commit 3125b7b7c301b8522e8c9cfb44f5d35d8b66214d \
  --base ../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4 \
  --model ../TurnItOver/models/Qwen3-VL-2B-Instruct \
  --out output/kimi-v4-increment-v1-reproduce \
  --report output/kimi-v4-increment-v1-reproduce-audit
```

远程运行根目录为 `/home/laq/experiments/kimi-v4-increment-v1`。为新复现选择空目录；脚本拒绝覆盖既有训练/评估输出。将本目录 `env-spec.json` 的环境构建到 `venv`，按 `ENVIRONMENT.md` 做完整验证。rsync 本地 `turnitover/` 与本轮 train/evaluate 脚本到远程 `code/`，保持相对目录。将同一原始模型完整复制到远程 `models/Qwen3-VL-2B-Instruct/`，比对每个文件的 SHA。将原 v4 的 manifest、train episodes/stopping、validation episodes/stopping 和 images 放入 `data/baseline/`；不复制 actions、private annotations 或 test JSONL。将生成的 `experiment-data/` 放入 `data/experiment/`。两份 data 中的图片均为目录内真实文件，不能用指向外部的软链接绕过 loader。

`run-remote.sh` 是实际使用的每组训练与评估命令。远程并行启动：

```bash
cd /home/laq/experiments/kimi-v4-increment-v1/code
nohup bash ../docs/run-remote.sh baseline 0 > ../output/baseline.console.log 2>&1 < /dev/null &
nohup bash ../docs/run-remote.sh experiment 1 > ../output/experiment.console.log 2>&1 < /dev/null &
```

每组从基础模型开始；第1617步后保存唯一 checkpoint 并运行相同 Validation。原 v4 evaluator 的逐例 raw 输出和 metrics 保留在 `output/{baseline,experiment}-eval/`。复制远程完整 output 到本地 `output/kimi-v4-increment-v1/remote-output/` 后，在本地追加严格评分（private annotations 只用于分层）：

```bash
python scripts/audit_kimi_v4_run.py \
  --root output/kimi-v4-increment-v1/remote-output \
  --report docs/evidence/kimi-v4-increment-v1
python scripts/score_kimi_v4_increment.py \
  --dataset ../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4 \
  --baseline output/kimi-v4-increment-v1/remote-output/baseline-eval \
  --experiment output/kimi-v4-increment-v1/remote-output/experiment-eval \
  --out docs/evidence/kimi-v4-increment-v1/comparison.json
python -m pytest tests/unit/test_kimi_increment_scoring.py tests/unit/test_oracle_training.py tests/unit/test_sft_diagnostics.py -q
```

`*-planned-order.jsonl` 是训练前由相同 shuffle 生成的计划；训练后的 loss.jsonl 必须逐步核验 id、epoch、objective 与计划一致。checkpoint 与大文件保存在本地 ignored output 目录及远程实验目录；Git 保存配置、来源/文件哈希、完整预测、对比与复现说明。无自动 push。
