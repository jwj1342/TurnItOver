# Kimi 教师诊断增量：v4 小规模公平对照

## 1. 数据是否成功接入 v4？

**成功接入 10/10 条：7 clean、3 joint_axis、0 part_offset。** 来源固定到实际 fetch 的 `origin/laq/sft-dataset` commit `3125b7b7c301b8522e8c9cfb44f5d35d8b66214d`，仅使用 `docs/evidence/kimi-easy-interactive-v2/sft/train.diagnosis.jsonl`。52 条 action 行及117条旧教师数据均未使用。

源文件 SHA-256 为 `21e8b2100e4f7af444fe21a2dd70f152c48425825704aad4ea338fc82cb984ec`。从该 commit 的 Git 对象导出不可变 snapshot，再逐条核验60个图像槽位的相对路径、图像 SHA、来源 episode/audit SHA、accepted 状态、独立检查器的定位/证据结论，以及原始 PartNet 身份。7个新监督原始资产的 source identity/content SHA、数据源版本与 model_id 均与原 v4 Train 对应记录一致；扩展后的 Train 与 Validation 在 source_asset_id、source_identity_sha256、source_content_sha256、split_group 四种键上均无交集。

没有视觉或证据协议阻塞。v4 多图接口可直接承载完整的 rest/mid/limit 参考图和真实交互历史；保留 reference/candidate 顺序、状态、hierarchy、joint_structure、证据引用及原始教师答案，仅将新增图片的文件路径前缀调整到实验目录内。真实 v4 formal terminal loader 与 Qwen processor 成功处理全部10条，所有用户/图像 token 均被 mask，assistant token 非空且未被 mask。远程实际训练环境也逐条复核；两地每条输入和 assistant token 数相同，远程真实 LoRA forward/backward、greedy generation 通过。

重要语义边界：新增 clean 的 `pass` 只针对输入明确声明的单关节运动 scope；原 v4 clean 的目标仍为 `uncertain`。未将局部通过改写成整物体无缺陷，也未改写答案或把隐藏 Gold 送入输入。源独立语义复核是另一 AI 的事后复核，结合既有人工资格审核 cohort 和检查器，不是本轮新增人工 Gold；教师 severity/confidence 未被当作独立校准标签。

证据与清单：`data-audit.json`、`used-kimi-rows.json`、`remote-smoke.json`、`fresh-environment-review.json`。

## 2. SFT 是否带来改善？

**只观察到微小的描述性净增益，尚不足以确认诊断能力稳定改善。** 两组均在远程 RTX 5090 D v2 上从同一基础模型重新训练；不是拿历史 checkpoint 与新运行比较。相同 seed 20261013、bf16/SDPA、LoRA rank8/alpha16/dropout0、q/k/v/o_proj、AdamW、学习率2e-4、1617 optimizer steps、相同 diagnosis/stopping 权重及相同 greedy/512-token 评估。

Baseline 为原 v4 的110条 terminal diagnosis +429条 stopping prefix，共539行；Experiment 为相同数据加10条新诊断，共549行。均只保存最后一步 checkpoint，无 Validation 模型选择。实际逐步 id/epoch/objective 全部与训练前计划一致：Baseline 330次诊断、1287次停止；Experiment 350次诊断、1267次停止，其中新 Kimi 出现30次（clean21、axis9）。总步数相同，但新增行改变 shuffle 及诊断/停止曝光比例，token/FLOPs 也不完全相同；不能把差异完全归因于教师的推理质量。

独立 Validation 为7个原始资产、28条原 v4 terminal 输入。下列按整行严格匹配计数，joint 按 `(defect_id,joint_id)`，parts 按 `(defect_id,joint_id,parts)`，evidence 按再加入 `evidence_steps` 的完整多重集合；完整严格准确率还要求 verdict status 正确。自然语言描述、severity/confidence 不作字符串相等评分。

| 指标 | Baseline | Experiment |
|---|---:|---:|
| 合法最终输出 |28/28|28/28|
| verdict status |28/28 (100%)|28/28 (100%)|
| defect_id |25/28 (89.3%)|25/28 (89.3%)|
| defect_id + joint_id |25/28 (89.3%)|25/28 (89.3%)|
| defect_id + joint_id + parts |17/28 (60.7%)|18/28 (64.3%)|
| 完整 finding + evidence_steps |17/28 (60.7%)|18/28 (64.3%)|
| 全项严格准确率 |17/28 (60.7%)|18/28 (64.3%)|
| finding precision / recall / F1 |0.421 / 0.421 / 0.421|0.474 / 0.474 / 0.474|
| unsupported findings |11/19|10/19|

按原始缺陷分层的严格字段计数如下。组合缺陷单列，不将其计入单独 axis/offset 的分母。

| 原始类别 | n | status B→E | defect_id B→E | joint B→E | parts B→E | evidence / 全项 B→E |
|---|---:|---:|---:|---:|---:|---:|
| clean |9|9→9|9→9|9→9|9→9|9→9|
| joint_axis |4|4→4|4→4|4→4|4→4|4→4|
| part_offset |5|5→5|3→3|3→3|1→1|1→1|
| axis + offset |10|10→10|9→9|9→9|3→4|3→4|

这里的 clean 正确指保持 `uncertain`，其空 findings/evidence 的正确不能解释为已学会验证 pass。组合缺陷继续沿用 v4 first-failure 目标，每例仅一个审核 finding，不评价找出所有注入缺陷。分层私有标签仅供离线评分，未进入模型输入。

### 改善与退化的具体内容

- 改善：`6ba47ca5a0c4c7519d9b88a6:verdict`（PM_102252）与 `463029325e71882436ea8e5d:verdict`（PM_1280），offset 部件从错误的 `link_2` 改成正确的 `link_1`。
- 退化：`25df51b717a0c0526562a5e0:verdict`（PM_1280），offset 部件从正确的 `link_2` 变成错误的 `link_1`。
- 其余25例全项正确与否不变。所有7例 finding 元组变化都只改变 offset 的 parts，均为编号减一（5例 link_2→link_1、2例 link_4→link_3）；没有 defect_id、joint_id 或 verdict status 的变化。
- 28例中没有任何一例改变 evidence_steps。evidence 联合严格匹配多1例来自部件匹配变化，不能据此声称证据引用或视觉使用能力增强。

这与部件命名偏好的变化相容，不能排除该解释。它不是“已证明 link_1 偏置导致变化”的因果结论。只有一个 seed、7个资产簇、4例纯 axis 和5例纯 offset；净增1例不构成稳健收益证据。新增数据仅3条 axis、无 offset；纯 axis 原已4/4，本轮没有显示额外 axis 收益，纯 offset 仍1/5。新增 easy 多状态协议本身没有独立 Validation 行，因此本轮只测向原 v4 协议的迁移，不能证明新协议泛化、完整物体 pass 验证或主动交互能力。

完整数据在 `comparison.json`、`paired-summary.json`、`finding-changes.json` 与两组 `*-eval/{metrics.json,predictions.jsonl}`。训练预算、逐步顺序和 checkpoint 哈希核对见 `run-audit.json`。

## 3. 下一步最值得做什么？

**选择：扩充可信 Kimi 诊断数据。** 决定性证据是新增监督0条 offset 正例，而 Validation 的19个非空目标中15个为 offset；本轮主要未解决的仍是 offset 类型/部件定位，纯 offset 严格准确率仅1/5。优先补充经过独立部件定位与证据审核的 offset 诊断、易混淆部件对照、不同部件编号和更多原始资产，避免继续按7 clean /3 axis 的比例重复扩充。

这是一项由数据覆盖缺口支持的下一步优先级判断，不是“更多数据一定有效”的结论；完整交互 SFT 也不直接解决当前固定观察下的定位失败。本轮到此结束，不启动新采集或补跑。

独立上下文的 result-to-claim 复核为 `partial`、confidence `medium`，见 `claim-review.json`。评审为 same-family / provisional，未运行完整 experiment-audit；确定性产物审核通过不替代语义质量认证。首次评审因代理额度失败，用户恢复后以相同 gpt-6-astra / ultra 路由成功重试，未降级模型。

## 保存与复现

- 固定配置：`configs/kimi-v4-increment-v1.json`；实际运行配置：本目录 `baseline-train/run.json`、`experiment-train/run.json`。
- 本地完整 checkpoint、逐步 loss 和运行日志：`output/kimi-v4-increment-v1/remote-output/{baseline,experiment}-train/`，checkpoint 为其中 `checkpoint/`。
- 本地来源 snapshot 与适配数据：`output/kimi-v4-increment-v1/{snapshot,experiment-data,kimi-diagnosis}/`。原 manifest 作为 v4 parent-export 元数据原样保留；本轮新增数据及实际训练行数以 `data-audit.json`、训练 run.json 和本轮固定配置为准。
- 远程完整产物：`/home/laq/experiments/kimi-v4-increment-v1/`（`laq@10.201.129.199`）。认证信息不写入文件。
- 基础模型的全部文件 SHA：`model-manifest.json`；两个 adapter 的 SHA：`run-audit.json`。
- 复现命令：`REPRODUCE.md`、`run-remote.sh`；环境版本与依赖：`env-spec.json`、`pip-freeze.txt`。
- 检查：相关本地测试14通过、2个因当前解释器缺 torch 跳过；真实 Qwen processor/mask、远程 forward/backward/generation、训练顺序/配置/checkpoint/配对输出审核均已实际通过。冻结 Test 未修改、未用于训练、评估或选择。
- Git 仅提交本轮代码、配置和可审阅证据；checkpoint 和大文件完整保存在本地 ignored output，未自动 push。既有未提交的 motion-grounding 工作不纳入本次提交。
