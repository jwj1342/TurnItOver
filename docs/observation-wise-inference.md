# Ordinary / Evidence 的逐 observation 推理

Ordinary 和 Evidence 共用 `infer_observationwise`：依次对 zero、midpoint、limit 的真实 `[reference, candidate]` 图像对独立推理，再确定性聚合。每次只包含当前图像对及相应公开 runtime 文本；没有 state 的公开 runtime 保留。训练数据、训练目标和 checkpoint 不由此模块修改。

聚合按 defect ID、部件 ID 集合、joint ID 去重，证据步骤映射回原 case。实际产生 finding 的 observation ID 存入 `findings_metadata.evidence_observations`；不同修复建议及其来源全部保留，冲突不会被后一次输出覆盖。推理和聚合不读取 Gold exposure records。

至少一个有效 finding 对应内部协议 `fail`（即 defective）；全部三个 observation 明确 pass 才能得到 case pass；其余情况 uncertain。无效输出和调用错误记录在 observation 日志中，不会升级为 pass。内部仍沿用现有 pass/fail/uncertain schema。

## Recognition

`scripts/evaluate_qwen_episode_sft.py` 的 diagnosis 默认采用 `--inference-mode observation-wise`，Ordinary/Evidence 使用同一入口及解码配置，仅更换 adapter。`--inference-mode sweep` 显式保留旧调用方式，stopping 不变。

```bash
python scripts/evaluate_qwen_episode_sft.py \
  --dataset /path/to/frozen/dataset \
  --dataset-manifest-sha256 MANIFEST_SHA256 \
  --model /path/to/Qwen3.5-2B \
  --adapter /path/to/ordinary-or-evidence-adapter \
  --out /path/to/new-validation-output \
  --split validation --task diagnosis --inference-mode observation-wise
```

`--withhold-observation zero|midpoint|limit` 可重复，移除发生在模型调用之前；若全部移除，零次调用并返回 uncertain。评测端根据 Gold 定义要移除哪些 observation，模型和聚合器只收到移除后的输入。每条 prediction 保存 observation raw/status/error/usage、聚合结果和证据来源。tokens 汇总所有实际 verifier 调用，不增加聚合 LLM 调用。

## Repair

`make_observation_verifier(acquire, model_call)` 返回现有 `run_iterative(..., verifier_call=...)` 接口的适配器。`acquire` 对当前候选程序重新取得真实 observation record 和图像路径；`model_call(record, images)` 与 Recognition 使用相同的单 pair 回调协议。适配器保存三次调用及 token usage，聚合 verdict、证据来源和建议冲突直接传给原有 Generator。独立检查、程序 ABI、生成协议和各 baseline 无需改变。

适配器要求提供项目对应的 fresh observation acquisition，不能把初始候选的旧 runtime 当成修改后证据。此变更验证的是推理组织、聚合和接线，不产生新的模型准确率或修复率实验结果。
