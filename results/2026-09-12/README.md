# 2026-09-12 实验测量快照

先读[当前研究状态](../../docs/research-status.md)了解问题、结论与限制。
本目录包含 124 份原始测量记录（约 270 KB），由 `scripts/export_research_snapshot.py` 按明确文件清单复制。
`snapshot.json` 保存每份文件的 SHA-256 与大小；复制过程不改写历史记录。

| 目录 | 内容 |
| --- | --- |
| `data/benchmarks/replicacad-v1/` | 数据资格与分组摘要 |
| `data/benchmarks/oracle-v1/` | 教师暴露、阈值、动作成本与重放验证 |
| `data/benchmarks/paired-v1/`、`oracle-action-distributions-v1/` | 训练导出规模、输入条件与歧义记录 |
| `output/replicacad-fk-v3/` | 原始 URDF 与浏览器的独立转换审计 |
| `output/qwen-*/` | 完整协议、主动策略、二图辨别和最小分类指标 |
| `output/repair-pilot-prepared-v2/` | 案例分层与准备输入哈希清单；被哈希的原文件未全部随附 |
| `output/repair-generator-sol-smoke-v1/` | 独立首条调用和修改前后审计 |
| `output/repair-sol-pilot-v1/` | 16 条修复轨迹、30 份原始响应与模型用量、逐轮审计和汇总 |

修复 `summary.json` 是执行汇总，`analysis.json` 将干净保持与损坏修复分开，并汇总返回费用。
每个案例/条件下的 `result.json` 保存分数轨迹与终止原因；`round-*/response.txt` 是实际模型补丁或停止响应；
`model.json` 保存模型标识、token、耗时及返回费用。`*.private.json` 是特权评估记录，不得当作普通模型输入。

这是测量快照，不是完整渲染重放包。资产、源码大数组、图片、权重与凭证不在其中；原记录中的
`data/`、`output/` 路径指执行时工作区，克隆后未必存在。历史清单中的“尚未完成”限制只描述
该次运行当时的范围，当前进展以研究状态文档为准。没有随附的原输入只能通过完整生成流程重建；
哈希提供核验依据，不能恢复原文件。

无需 API 即可核对所有文件：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('results/2026-09-12')
for name, record in json.loads((root / 'snapshot.json').read_text())['files'].items():
    data = (root / name).read_bytes()
    assert len(data) == record['bytes']
    assert hashlib.sha256(data).hexdigest() == record['sha256'], name
print('All measurement hashes verified')
PY
```

在已配置的 Python 环境中重新汇总修复指标：

```bash
python scripts/summarize_repair_experiment.py results/2026-09-12/output/repair-sol-pilot-v1
```
