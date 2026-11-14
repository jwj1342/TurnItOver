# RP29 v2.1 Validation 批量修复轨迹采集

`prepare_rp29_full_repair.py` 冻结 v2.1 Validation 的全部 109 defective cases，
排除 11 clean；不读取 Test。每例清单绑定程序、reference 图与真实重放配方。
24 个缺失 supplementary 配方已从原 observation-plan／时间合同恢复；出处与哈希
保存在 `recipe_provenance`。3 个 joint_range 案例须按 `commands_by_role` 重放，
3 个 temporal 案例保留 `time_sequence`，不把同一数值命令强加给两个程序。

Train 600 全部是 defective。pixel 校准使用这 600 例和 600 个明确标记的
reference-self 合成 clean 对照，未使用 Validation 调参。当前阈值为 0；
它不代表真实 clean 泛化校准，像素相同也不代表 runtime／物理正确。

先进行不调用模型的输入预检：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_rp29_repair_batch.py \
  --manifest ../.run-artifacts/rp29-table1-full-20261114/manifest.json \
  --native-root ../TurnItOver-taxonomy-900 \
  --output ../.run-artifacts/rp29-table1-kimi-full \
  --preflight
```

明确选择未评分轨迹采集后执行：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_rp29_repair_batch.py \
  --manifest ../.run-artifacts/rp29-table1-full-20261114/manifest.json \
  --native-root ../TurnItOver-taxonomy-900 \
  --output ../.run-artifacts/rp29-table1-kimi-full \
  --env-file /path/outside/repository/kimi.env \
  --conditions none pixel --rounds 3 --collect-unscored
```

两组各运行一次，共 218 条轨迹。none 固定消耗三轮，pixel 可由自己的 judge
或 generator stop 提前结束。私有标签、reference 源码、归档 candidate runtime
不进入模型；none 只获得 reference 图、当前源码、公开任务和公开预算。
每个新源码哈希重新执行原生浏览器，pixel 使用新采集的三态配对图。

逐例错误会保留并继续后续案例；重启同一输出目录会跳过已有结果和中断案例，
不自动重试可能已计费的请求。输入、校准、配置或执行代码变化时拒绝续跑。
`config.json` 记录单次重复、模型、轮数和 seed 未请求／未核实状态。
`progress.json`／`summary.json` 记录完成数、错误、调用与提供方报告的 token 消耗。

**本入口未实现完整 29 类独立评分。** `status=complete` 仅表示轨迹采集完成；
Success、Resolved、Needs motion／Visible at rest 和 Cost to success 均不能据此填表，
相关评分保持 null。pixel judge pass 只控制自身停止，不充当独立成功标签。

验证记录：2026-11-14，109/109 个原始 defective 程序均完成真实原生浏览器
三态重放，0 API 调用、未用 GPU。记录在
`../.run-artifacts/rp29-repair-code-check-20261114/all109/summary.json`。
验证环境为 Playwright 1.58.0／已缓存 Chromium；可使用
`uv run --with playwright==1.58.0 python ...` 复用该版本，避免浏览器版本不匹配。
该记录只证明初始输入可执行，不代表模型产生的任意 patch 均可执行。
