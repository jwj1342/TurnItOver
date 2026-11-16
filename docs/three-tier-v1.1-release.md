# Three-Tier-900 SFT v1.1 预发布

本版是 `three-tier-900-sft-v1`（5 类缺陷，Easy / Medium / Hard）的标签修订，发布 tag 为 `three-tier-900-sft-v1.1-pre.1`。它不是 120 例 `benchmark/v1` 的重命名，也不是 29 类 RP29 v2.1；不修改这些版本的历史结果。

## 本次更新

- 450 个原始 clean 案例的诊断目标从 `uncertain` 改为 `pass`；配套停止文件从 `continue` 改为 `stop/pass`。其中 Train 300、Validation 60、Test 90。
- clean 准入依据是已发布参考来源、参考与候选程序哈希相同、原冻结完整观测和图像哈希校验；Validation 60 clean 另有 #16 人工逐例对账确认。没有把缺陷标签为空单独当成充分依据，也不宣称 Train/Test 在本次新增了逐例人工盲审。
- pass 仅表示规定的 Structure/Kinematics 固定观测任务范围内通过，不表示任意视角、任意状态、几何或物理属性均正确。
- 每条记录补 `is_clean`、`label_policy`、`admission`；clean 增加绑定案例、程序和完整输入哈希的私有 `pass_certificate`。这些字段不进入模型 user prompt。
- 接入 120 条人工对账记录，保留原始判断、备注、审核者、时间和原始难度。13 个注入后人工判 clean、3 个 invalid 均保留原注入标签和旧目标，标为 `quarantined_pending_adjudication`；默认加载器排除，原始文件中不删除、不改成 pass，也不自动解释为不可见。
- Ordinary 继续用全部注入诊断，Evidence 继续用当前输入支持的诊断与引用。无缺陷暴露的 defective case 在 Evidence 中仍为 uncertain；两种条件使用完全相同的公开输入。
- 配套运行代码允许经过认证的 pass，拒绝无认证、换案例、换程序或缩短观测后的 pass；补齐 clean 误报、clean pass 率、真实 defective false-pass 和 uncertain 指标。所有模型评测统一以 Evidence 目标为比较标准，不能各用各自的训练目标打分。

## 数量与评测分母

| Split | 原始案例 | 默认准入 | 待裁决 | 原始 clean | 准入 Evidence pass/fail/uncertain |
|---|---:|---:|---:|---:|---|
| Train | 600 | 600 | 0 | 300 | 300 / 279 / 21 |
| Validation | 120 | 104 | 16 | 60 | 60 / 44 / 0 |
| Test | 180 | 180 | 0 | 90 | 90 / 90 / 0 |

原划分、案例 ID、物体身份及 6600 张图像均保留；Train 281、Validation 60、Test 90 个来源物体，无跨划分物体泄漏。原始每层 300 例，分别 Train 200 / Validation 40 / Test 60；默认 Validation 的 Easy/Medium/Hard 分母为 **39/35/30**，合计 104，仍覆盖原 60 个来源物体。隔离是本版预发布的保守准入处理，不是对那 16 例原因的人工裁决。

Ordinary Train 为 300 pass / 300 fail，Evidence Train 为 300 pass / 279 fail / 21 uncertain。三态输出接口保持一致，不要求不同监督策略的目标分布相同。Base 不训练；Ordinary/Evidence 各从同一底座独立训练。

默认 Validation 没有已准入的 uncertain 正例，因此该集的 uncertain recall 为 null，不能以此声称拒答能力已验证。新训练标签会改变模型学习，旧 v1 checkpoint 和分数不能重新命名为 v1.1 结果。若以后裁决 16 例或补充 withheld-evidence 评测，应另行冻结版本和协议，不依据模型表现选择保留案例。

## 文件与加载

- `three-tier-900-sft-v1.1.zip`：900 例完整导出、原图、私有注入标签、标签变更清单、quarantine 清单和人工对账记录。
- 训练与评测继续使用 #10 的 `laq/three-tier-900-sft-export` 分支。本版在 `88ac0df359a454bfea8c4a3fd3e920fec8b24381` 基础上增加 v1.1 标签兼容，数据版本对应的固定代码 commit 见 [#10 更新评论](https://github.com/jwj1342/TurnItOver/issues/10#issuecomment-6904414090)。训练入口、Base/Ordinary/Evidence 对照和原环境不变。
- `three-tier-900-sft-v1.1-runtime.zip` 是同一套配套运行代码的离线发布快照，可以作为校验或离线备份；不要求用附件替代 Git 分支。更新前的 900 诊断加载器拒绝 pass，此限制不适用于本次更新后的代码，也不适用于原本支持 pass 的 120 例二分类代码。
- `release-manifest.json`、`SHA256SUMS.txt`：来源、数据 manifest 和附件哈希。
- `verification.json`、`unit-tests.txt`：发布校验记录，属于数据与程序测试，**不是模型实验结果**。

优先在已有代码仓库检出 #10 评论固定的 v1.1 兼容 commit，并使用私有 Release 的数据附件。完整下载发布附件时执行 `sha256sum -c SHA256SUMS.txt`，数据解压到新目录。离线使用时也可在解压后的 `three-tier-900-sft-v1.1-runtime` 中运行。两种方式均沿用现有兼容 Python 3.12 环境和模型依赖，不自动安装或修改环境。环境依赖见 `pyproject.toml` 与 `requirements-qwen.txt`。

训练入口仍为 `scripts/train_qwen_episode_sft.py --task diagnosis --supervision ordinary|evidence`，数据参数指向解压后的 `three-tier-900-sft-v1.1`，传入本版数据 `manifest.json` 的 SHA-256，其他训练参数按冻结实验配置提供。评测入口为 `scripts/evaluate_qwen_episode_sft.py`，Base 和所有 SFT 均使用相同数据与默认准入规则。评测报告同时记录原始数、排除数和实际分母。配套 `.stopping.jsonl` 为兼容历史工具保留；本次不启动独立停止策略训练或主动取证实验。

重建步骤需要旧私有候选 ZIP、其 snapshot、120 条对账 JSONL 和对应对账摘要；运行 `scripts/build_three_tier_v11.py --help` 查看参数。必须匹配旧 archive SHA256 `185d2d6ba7ca2b6d2fb7426d407e50c9682a8632e3a827f18502cab9c309e2e0` 和旧 manifest SHA256 `6d0fc5344810396e9454e83c58572f9a9ecdcae4cb53a477f59e9dd541fc6c25`。在仓库会话目录保存了本版构建脚本、代码补丁和审核来源哈希。

## 检查与解释边界

本次验证所有归档成员、6600 张图像、1800 条 diagnosis/stopping 记录的公开输入保持不变，核对原始物体隔离，并用真实加载器读取 Train 600 / Validation 104。检查 Ordinary/Evidence 的公开输入相同、clean 目标相同、停止目标相同，以及合成“全部 pass”“全部 uncertain”和正确目标回放的评分行为。完整非浏览器测试为 **229 passed、5 skipped、19 deselected**；跳过和未选择项不计通过。未运行训练、模型推理、浏览器渲染或修复实验。

Evidence 的支持标签继续来自旧版 `counterfactual_pixel_exposure`，不是新增的逐 finding 人工语义 grounding。旧版复合案例的曝光生成只将像素信号归给指定首个标签，本次未补造其他标签的可见性。人工确认“有问题”也不等同于逐缺陷类型、部件和引用步骤均已确认。

Test 仅为机械补标、完整性校验和打包读取，没有进行模型评测、阈值选择、训练或案例选择。原始 test 划分继续保留。

这是所有者授权的实验性标签预发布，不是已完成全部人工审核的正式 benchmark，也不证明修复停止能力已达标。现有 clean 仍是参考自身对照，未新增正常外观变化，不能据此声称解决像素捷径。

数据、图片、私有标签及审核回件只发布在 `mmk1209/TIO-datasets` 私有仓库及其 Release。遵循原 PartNet-Mobility/ShapeNet 非商业研究使用与访问限制，不得转存到公开仓库或未经授权的服务。

关联：[训练 #10](https://github.com/jwj1342/TurnItOver/issues/10)、[人工审核 #16](https://github.com/jwj1342/TurnItOver/issues/16)。RP29 v2.1 的独立进展仍见 [#17](https://github.com/jwj1342/TurnItOver/issues/17)。
