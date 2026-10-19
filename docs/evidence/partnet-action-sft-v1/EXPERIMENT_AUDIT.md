# PartNet Action-only SFT 实验完整性审计

- 审计者：fresh Codex 同模型家族审计 agent `/root/final_integrity`；直接读取代码与原始工件，未采信执行者的 PASS 结论。
- `review_independence: same-family`；`acceptance_status: provisional`。
- **总体：WARN；未发现阻断当前限定结论的完整性问题。** 不能把本次结果升级为稳定主动策略、视觉语义能力或缺陷诊断成功。
- 审计只写入本报告与同名 JSON；未重训、推理、修改实验数据、重新渲染或打开 frozen Test。

## 独立验证结果

独立脚本从旧 Train/Validation action JSONL 重建可见前缀键和动作频次，再与概率导出逐组比较：Train 660 行 → 496 组/500 变体，Validation 168 行 → 121 组/122 变体；3/1 个冲突组的全部经验概率均完整保留。868 项来源/图像 SHA 检查通过，Train/Validation 的物体、来源身份、内容 SHA、参考 SHA 与 exact-prefix 无交集；原始正式 catalog 为 35 个 Train/Validation 物体。候选腐蚀标签来自原 raw `record.json`，Validation 为 28 案例、7 物体、19 缺陷案例、29 项注入缺陷。

1488 条训练更新连续，三个 epoch 各覆盖全部 496 组一次；每条 loss 等于完整组内加权和。在线 epoch 均值重算为 `0.0627456525 / 0.0159638167 / 0.0143097411`。adapter 有 224 个 tensor、1,605,632 个元素，所有 tensor 有限且非零；其 SHA 为 `0fc95434c665b4ee587c5d8cf930408d28e3c6cb909057c08e7eeae21745d346`，与训练记录和真实 SFT server ready 行一致。

正式原始目录仅为 `output/partnet-action-sft-v1/baselines-run3/` 和 `action-sft/`。审计遍历全部 **336 episodes、1344 次耗预算观察、672 次 Active 动作回复、849 次最终诊断调用和1680 张 PNG**。每条 Active 原始回复解析结果与实际执行动作相同；所有动作请求的 history、remaining 和 image_order 与已取得前缀精确一致，无未来图像。最终诊断请求包含实际完整轨迹，且同样不含私有标签。所有正式调用只有一种动作阶段协议、一种 final 协议。全部 PNG SHA 和参考图像字节匹配。

849 次 final 调用中，172 次成功形成各自 episode 的有效最终 verdict；670 次失败于字段/JSON/引用校验，7 次截断。逐条重解析证实没有从失败输出择优抽取合法片段或剔除失败案例。有效输出不代表语义正确。1,234 条学习评估记录的生成动作、合法性、经验质量、支持动作率、分组均值和 initial/after-initial 分层均重算一致。NLL 仅验证保存数值的聚合与已审查计算路径，本审计未另开 GPU 重算 logits。

原 Fixed/Random witness planner 的 168 条轨迹由原物体关节限位和 seed 逐条重建，与已执行动作一致；离线特权 Oracle 重新规划得到 K2=27/29、K4=29/29、K6=29/29。Stopped `baselines/` 保留18条 episode、`baselines-run2/` 保留15条，均无 complete metrics，正式 merge 没有纳入。原始训练数据和 checkpoint 未因协议修复重训。

| K | 策略 | 曝光/29 | 任一证据/19 | 全缺陷曝光/19 | 诊断正确/28 | 有效最终输出/28 |
|---|---|---:|---:|---:|---:|---:|
| 2 | zero_shot | 10 | 10 | 2 | 0 | 10 |
| 2 | action_sft | 17 | 14 | 9 | 1 | 15 |
| 2 | fixed | 15 | 13 | 6 | 2 | 8 |
| 2 | random | 13 | 12 | 4 | 1 | 10 |
| 4 | zero_shot | 10 | 10 | 2 | 1 | 13 |
| 4 | action_sft | 19 | 15 | 10 | 4 | 16 |
| 4 | fixed | 16 | 14 | 7 | 5 | 21 |
| 4 | random | 17 | 14 | 8 | 3 | 11 |
| 6 | zero_shot | 10 | 10 | 2 | 4 | 17 |
| 6 | action_sft | 22 | 16 | 13 | 9 | 25 |
| 6 | fixed | 25 | 17 | 15 | 2 | 9 |
| 6 | random | 19 | 16 | 9 | 6 | 17 |

上述每一格均由原始 episode/verdict、数据集腐蚀标签和 `evaluation.private.json` 的数值信号重新计算。像素阈值布尔值与曝光集合也逐条重算；本审计未重新渲染未存档的 counterfactual PNG，因此不把这些数值一致性检查称为独立像素重放。

## A. Ground Truth Provenance — PASS

监督来自已有特权教师频次，真实诊断标签来自原数据集注入腐蚀；没有把模型输出充作 GT。概率聚合不泄露成员身份到模型。环境可以私下实例化候选，这与模型收到私有几何/标签是不同边界。所有正式请求字段和实际图像均验证为公开观察前缀。训练只载入 Train，学习评估只载入 Train/Validation，闭环按 Validation IDs 选择；raw shard 的 Train/Validation 元数据用于实例化，未访问独立 frozen Test。

证据：`turnitover/oracle/ambiguity.py:13-29,59-76`；`turnitover/oracle/supervision.py:22-35`；`scripts/evaluate_action_sft_closed_loop.py:63-79,102-125,146-195`；`turnitover/models/action_qwen.py:49-59`；`scripts/action_sft_model_server.py:22-35`。原始具体样例：`output/partnet-action-sft-v1/action-sft/k2/action_sft/6ba47ca5a0c4c7519d9b88a6/episode.json:1` 及同目录 `evaluation.private.json:1`、`action_calls/001/prompt.txt:1`。

## B. Score Normalization — PASS

assistant-only mask 覆盖全部输入前缀，CE 在每个答案内部按有效 assistant token 平均，组内按完整经验概率加权，每组更新一次。未发现以模型自身最大值/均值归一化分数。曝光分母固定为29项注入缺陷，缺陷案例证据分母为19；诊断分母为全部28案例，失败输出不移除。整动作类别经验熵与可变答案 token 平均 CE 不同单位，不能以熵数值指控 loss 低于不可约下界；PROTOCOL 已说明此限制。

证据：`turnitover/oracle/training.py:19-57`；`scripts/train_qwen_action_sft.py:77-95`；`scripts/evaluate_action_sft_learning.py:43-74`；`scripts/action_sft_metrics.py:17-28`；`docs/evidence/partnet-action-sft-v1/PROTOCOL.md:17-21`。

## C. Real Results / Provenance — WARN

实际训练、adapter 保存和 SFT 服务重载、闭环模型原始回复、PNG、学习评估均存在且数字吻合。训练配置 rank4/alpha8/LR2e-4/seed20261017/epochs3；最终 adapter 不是空权重。正式评测未启用 fake diagnoser。

局限：`base-server-v2.log:2` 的旧 ready 仅包含 host/port，没有当时的基座权重、环境或代码 SHA；当前复现脚本不能追补历史运行证明。代码分支、请求协议和配置路径支持同一底座诊断方案，SFT ready 提供实际 adapter/base-config 身份，但不能声称已独立验证旧 base-only 服务的历史权重字节。`run_config` 也只哈希部分入口源码，未给所有传递依赖的当时快照。本 JSON 中的源码 SHA 固定本次审查版本，不追溯证明未记录的历史版本。

证据：`docs/evidence/partnet-action-sft-v1/train-run.json:2-18,616-625`；`output/partnet-action-sft-v1/sft-server.log:2`；`output/partnet-action-sft-v1/base-server-v2.log:2`；`turnitover/models/action_qwen.py:16-32`；`scripts/action_sft_model_server.py:23-28`；`docs/evidence/partnet-action-sft-v1/learning/metrics.json:4-35`。

## D. Dead Code — PASS

概率分组、训练 loss、实际模型 generate、动作执行、后验曝光和汇总函数均处于已记录运行路径，结果键与原始产物一致。EngineeringModel 只在明确 smoke 开关开启时选择；两次正式 config 均为 false。模型元数据均指向 Qwen3-VL-2B-Instruct。原 Fixed/Random、Loop、final prompt 未见相对当前 HEAD 的修改，168条控制轨迹能重建；此处“未修改”仅指已检查的本地 Git 差异，不替代运行时全依赖快照。

证据：`scripts/evaluate_action_sft_closed_loop.py:95-98,122-145,159-174,196-206`；`scripts/action_sft_metrics.py:46-74`；`turnitover/models/local_qwen.py:29-45`。

## E. Scope / Claim Boundaries — WARN

有效独立样本为7物体，28案例和三个 K 不能当作84个独立样本。单次训练 seed；无图像消融，不证明视觉语义依赖。K2/K4 相对 Fixed 的曝光增量分别是2/3项，**全部来自 PM_1280**；其余6个物体增量均为0。K6 相对 Fixed 减少3项，不能声称稳定优于强固定路径。

所有四策略、所有 K 的缺陷案例 exact diagnosis 均为 **0/19**。SFT 的 K6=9/28 完全来自9个 clean 案例；不能称作缺陷诊断恢复。exact diagnosis 仅比较 defect-ID 集合和 pass/fail 状态，不覆盖部件或关节定位、解释或语义证据。原 Fixed/Random 使用环境关节限位去重/恢复，这属于保留的特权结构化控制，不能称与模型完全相同的信息访问。

原 prompt 允许 verdict，而动作解析器禁止 verdict；正式闭环为两个 Active 同加 acquisition-only 指令，并完整排除早期失败运行。672次实际调用的动作协议一致，未发现只给SFT更有利指令或按成绩换checkpoint。它属于已披露的协议修复；原前缀学习评估保留旧 prompt，因此底座原前缀自由生成的低合法率不能解释成纯动作选择能力低，不能直接与修复后闭环合法率横比。

证据：`docs/evidence/partnet-action-sft-v1/REPORT.md:55-77`；`scripts/evaluate_action_sft_closed_loop.py:163-168`；`turnitover/verifier/action_only.py:16-20,87-90`；`turnitover/oracle/witnesses.py:21-35,43-51`；`scripts/action_sft_metrics.py:65-73`。

## F. Evaluation Type — WARN / simulation_only + dataset real_gt

真实 Qwen 推理与真实 Chromium 环境执行，场景资产为 PartNet，错误由数据集程序注入。主评估类型为 **simulation_only**；诊断标签属于该模拟 benchmark 提供的 **real_gt**。曝光是后验反事实像素代理，动作匹配是特权教师经验模仿；均不是人类审核、现实物体缺陷诊断或语义视觉能力证明。没有模型生成参考答案冒充真实 GT 的问题。

证据：`turnitover/oracle/evidence.py:14-26,33-45,63-71`；`scripts/evaluate_action_sft_closed_loop.py:100-159`；`scripts/evaluate_action_sft_learning.py:19-28`。

## Claim Impact / 后续处理

- “小模型能拟合现有经验动作监督”：支持，但仅限现有频次目标；不能变成最优策略或视觉理解声明。
- “K2/K4 比原Fixed取得更多像素曝光”：数字支持，必须同时写出PM_1280单物体集中、n=7和K6负结果。
- “稳定更优主动策略 / 恢复了缺陷诊断”：不支持。
- 当前报告已保留上述限定和负结果，无需为改善排名重训、改条件或追加本轮推理。未来另行授权的运行应在每个服务启动时保存权重/环境/代码SHA，并存档counterfactual评分图像。

JSON保存本次直接读取文件的SHA、全部独立重算计数、各项证据与声明影响。审计为same-family provisional，不构成cross-family独立接受。
