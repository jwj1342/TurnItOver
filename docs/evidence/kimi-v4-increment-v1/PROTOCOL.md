# Kimi 增量诊断：预先固定的比较协议

本次只做一组配对对照，不搜索超参数、不用 Validation 选 checkpoint。两组从相同原始 Qwen3-VL-2B-Instruct 权重开始，以 v4 的 `train_qwen_episode_sft.py`、`load_multitask_training_examples`、`tokenize_example` 训练。只增加可选 `--max-steps`，未改训练目标、优化器、LoRA 或 token mask。

来源固定为实际 fetch 的 `origin/laq/sft-dataset` commit `3125b7b7c301b8522e8c9cfb44f5d35d8b66214d`，只用该版本的 `docs/evidence/kimi-easy-interactive-v2/sft/train.diagnosis.jsonl`。snapshot 保存该目录上层的原始审核链；动作行、117 条旧教师数据和私有标签均不送入训练 loader。用于训练的新行及图片清单、SHA、输入与答案摘要见 `data-audit.json` 和 `used-kimi-rows.json`。

多状态参考图和交互历史完整保留。新增图像仍按原 `image_order` 依次输入，每个 reference 保留 view/joint/detent，candidate 状态由因果 action/history 恢复。hierarchy 和 joint_structure 是公共输入，axis parts 是直接移动子链接。教师的单关节 scope、完整 rest/mid/limit 观察及 `pass` 条件也完整保留，不能将局部 pass 解释为整物体全项通过。v4 的旧行仍保留原单图参考和原监督；不把新数据裁成单张参考，不重写教师答案。

数据审核先核验路径、60 个图像槽位的 SHA、来源 episode/audit SHA、accepted 标记、独立定位与证据检查，再核对原始 PartNet source_asset_id、source_identity_sha256、source_content_sha256、split_group。训练集合与 Validation 在四种键上均无交集。来源独立语义复核明确是另一 AI 事后复核，并非新增人工 Gold；原始 cohort 有既有人工资格审核，置信度和严重度仍是未校准的教师估计。

预算固定为 1617 optimizer steps、seed 20261013、学习率 2e-4、rank 8、alpha 16、dropout 0、q/k/v/o_proj，diagnosis 权重 1、stopping 权重 110/429。每样本 assistant-token mean CE，batch size 1。Baseline 539 行完整 3 epochs；Experiment 549 行，到第 1617 步截断。两组每一步预算相同，但 token/FLOPs 不完全相同，额外诊断改变 shuffle 和诊断/停止曝光比例；实际顺序须与 planned-order 核验。

主评估复用 `evaluate_qwen_episode_sft.py`，相同28个独立 Validation terminal episodes、7个原始资产、greedy、max_new_tokens 512。运行后追加严格评分：verdict status、defect_id、joint_id、parts、evidence_steps 逐行多重集合等价（不忽略重复 finding）；strict_all 要求 status 与所有完整 finding/evidence 元组均相同。原 evaluator 的合法证据子集指标另行保留，不能替代严格 evidence 准确率。clean 的空 finding/evidence 匹配不等于验证过 pass；v4 clean 目标仍为 uncertain。原始 private annotations 仅在离线评分时划分 clean / axis / offset / axis+offset，绝不进入模型输入。

报告必须列出单缺陷类别及组合缺陷，不把组合例混为纯 axis 或纯 offset。逐案例配对改善/退化和按原始资产的范围同时保存。此评估测量向原 v4 协议的迁移；新增 easy 多状态协议没有独立 Validation 行，因此不能估计该协议上的 held-out pass/诊断表现。只一个 seed、仅3条 axis 正例、0条 offset 正例，不作显著性、跨协议泛化、解决 offset 或完整交互能力的结论。冻结 Test 不用于训练、评分或选择。
