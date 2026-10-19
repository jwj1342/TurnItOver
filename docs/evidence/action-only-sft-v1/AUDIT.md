# Action-only SFT 数据审计（2026-10-19）

当前仅完成历史 manifest 与实现的审计。指定的 `oracle-action-distributions-v1`
原始动作 JSONL、图片、组成员表、ReplicaCAD raw/prepared 与资产没有在本机找到；
远端检索结果由主执行记录保存。历史 `/scratch/jwj/TurnItOver` 路径不存在。
因此不能核验逐行来源、图片 SHA、组内权重或执行真实训练与闭环，本文件没有本轮实验成绩。
结构化事实与 SHA 见 [data-audit.json](data-audit.json)。

## 来源与物体隔离

`results/2026-09-12/data/benchmarks/oracle-action-distributions-v1/manifest.json`
声明源为 `oracle-v1`，其 `source_manifest_sha256` 与本地历史源 manifest 完全一致。
这只验证 manifest 链接，不能代替缺失的数据内容校验。

历史 splits 的单位是 `split_group`。Train 包含 cabinet、chest_of_drawers、
kitchen_counter、fridge 四个 ReplicaCAD 资产；Validation 仅 kitchen_cupboards
一个资产。历史 benchmark 声明 Train 64、Validation 16 个腐蚀案例；16 个案例
不能计成 16 个独立 Validation 物体。清单层面的 Train/Validation 资产组交集为空，
原始记录身份与物体内容 SHA 的隔离尚无法重新验证。本轮未读取 Test 样本。

## 可见信息、冲突与损失

`supervision.action_examples` 构造 `当前参考 + history[:step] + remaining → action`，
下一步图片和私有标签没有加入 messages。`ambiguity.observable_key` 用图片字节
SHA 替换路径，连同协议、上下文、预算和既往观察形成键，且各 split 独立分组。
原 manifest 声明 Train 266 组、275 条带权目标，Validation 64 组、65 条带权目标。
Train 四个冲突组涉及 30 条源记录，分别有 2/6/2/3 种动作；Validation 一个冲突组
涉及六条源记录、两种动作。不能任取一个动作作为唯一 Gold。

动作概率是特权教师在相同可见前缀下的经验频率，不是可见信息下最优策略。
即使单例也可能无法从可见信息辨识。恢复数据后应使用 `load_action_groups`
整组加载，保留 `loss_weight=count/group_size`，沿用 `tokenize_example` 的
assistant-only masking 和图像处理，以 `weighted_assistant_loss` 计算组内加权
目标。应记录加权 loss、组内动作分布概率与合法动作率；逐条硬动作匹配不足以证明
动作监督可学习。目前缺失目标行，无法计算冲突条件熵、硬匹配可达上限或验证权重。

## 预算与闭环比较边界

历史 `oracle-v1` 的暴露曲线汇总了全部 split，没有独立 Validation 曲线；
本次不将该汇总作为 Validation 的 K=2/4/6 上界，也不从中推断策略是否可区分。
恢复既存 Validation annotations 后，才能在固定案例上审计各预算的暴露空间和
Fixed/Random/Oracle 差距。像素反事实暴露只是证据代理，必须与冻结最终诊断器
准确率独立报告，不能称为语义识别能力。

原监督的 Fixed/Random 来自 `oracle.witnesses.plan`，按 witness 顺序生成真实收费
动作，包含关节恢复和相机切换；`verifier.policies.action_plan` 是另一套已有基线。
本轮应明确沿用哪套，并保持四种策略允许动作、状态规则和初始参考一致。
`evaluate_active_qwen.replay_images` 已支持对真实历史后验反事实评分；私有标签与
Oracle 只能在策略结束后离线计分，不能进入推理调用。最终诊断器对所有策略必须
冻结且相同。Checkpoint、训练 loss、逐步模型响应/动作/图片、配置、数据 SHA 和
逐物体差异均须来自实际运行。当前“能否学会”和“能否改善取证”均未验证；数据
缺失不是可学习性负结果，也不能通过改用其他数据或扩大数据制造改善。
