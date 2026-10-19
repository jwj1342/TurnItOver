# PartNet Action-only SFT：预先确定的数据与比较协议

本轮获用户授权，使用现存 `../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4`，不处理 ReplicaCAD，不新增物体、Oracle 轨迹、诊断训练或停止训练。目录名中的 v4 指已有监督版本，不代表本轮重新训练 v4。

## 数据边界与来源

- Train：28 独立 PartNet 物体、110 案例、660 条既有动作；Validation：7 独立物体、28 案例、168 条动作。Validation 含 19 缺陷案例、9 clean 案例。
- 逐源记录检查 `data_identity`，Train/Validation 的 asset ID、来源 identity、identity SHA、content SHA、reference SHA、参考图像 SHA 和 exact public prefix 均无交叉。完整清单与 SHA 在 `data-audit.json`。
- 验证 828 个输入图像路径的所有声明 SHA（558 个不同 SHA）；原目录和既有数据不修改。动作消息只含当前参考、已发生观察、上下文、taxonomy 和预算，没有当前目标动作之后的图像，也没有私有缺陷字段。历史已去除 `render_ms`。
- 只读取 `train.actions.jsonl`、`validation.actions.jsonl` 和来源 manifest。原 formal catalog 和私有 annotations 仅有 Train/Validation；私有 annotations 用于离线预算审计，不进入模型。`test.*` 只检查文件大小，均为 0，从未打开；不访问独立冻结 Test。
- 原 `actions` 每案例 6 条，仍为完整 K=6 Oracle 轨迹；`first_failure` 的截断属于 episodes/stopping 导出。本轮严格复用动作流，不把终止标签混进动作训练。

## 可见输入冲突与训练目标

复用 `export_distributions`，在每个 split 内以协议文本、完整公共 JSON 和图像字节 SHA 分组，图像路径归一成 SHA，训练保留代表记录的原消息和图像。Train 得 496 组/500 动作变体，Validation 得 121 组/122 变体。

Train 有 3 个冲突组、10 条受影响源动作；Validation 有 1 个冲突组、2 条受影响源动作，冲突均在已取得首张观察之后。权重分别保留组内频率（包括 3:1、1:1:1、2:1 和 Validation 1:1），绝不任选唯一 Gold。私有分组来源映射不传给模型。所有案例首动作都是 front，预测首动作不能证明主动选择能力。

整动作类别经验熵的组平均 Train 为 0.004632 nats，Validation 为 0.005728 nats；确定性预测在本组样本频率上的 top-1 上限分别为 99.394% 和 99.405%。上述熵不是按可变长度答案 token 平均的训练 NLL 下界。源记录加权的 top-1 上限也不能与等权组评测直接比较；等权组的 majority-mass 上限 Train 为 99.748%，Validation 为 99.587%。这只是有限样本 exact-prefix 冲突审计：未发现 exact 冲突不能证明单例动作可由可见输入识别，且特权教师频率不等于可见信息下的最优策略。

沿用现有 Qwen3-VL-2B-Instruct Action-only 脚本及图像处理、JSON action schema、assistant-only mask。rank=4、alpha=8、LR=2e-4、seed=20261017、3 epochs。每组各变体按 `loss_weight` 进行 assistant token mean CE 加权，每组一次优化更新，共 1488 更新；记录逐变体 loss。不同组等权，不将源组频次重新乘入。只训练 Train，不据 Validation 调参或选 checkpoint。

## 预算区分度：仅已有轨迹的离线前缀审计

下表分母为 19 个 Validation 缺陷案例，数值为预算内至少曝光一个私有缺陷的案例数。使用既有 K=6 轨迹的 K 步前缀；没有生成新 Oracle，不能作为本轮真实闭环结果，也不是重新按 K 规划的路径。三种策略都固定首步 front，首步已覆盖 10/19。

| K | 已有 Oracle 前缀 | 已有 Fixed 前缀 | 已有 Random 前缀 |
|---|---:|---:|---:|
| 2 | 19/19 | 13/19 | 13/19 |
| 4 | 19/19 | 13/19 | 14/19 |
| 6 | 19/19 | 17/19 | 16/19 |

K=2/4 有区分空间，K=6 部分饱和，不能把较大预算下无增益直接归因于训练无效。K=2/4 相对于 K=6 监督是剩余预算与历史组合的分布偏移，单独报告，不据结果改变 K 或监督。

## 真实闭环比较与解释边界

相同 7 个 Validation 物体、28 案例、参考、合法动作、状态持久规则与 K=2/4/6，比较 Zero-shot Active、Action SFT Active、原 Fixed、原 Random。模型每步实际决定动作并由 Chromium 获得观察。各策略使用同一个冻结 base Qwen 诊断器（关闭 Action adapter）；不重训 v4/v5。保存提示词、输出、action、图像、解析失败、观察成本和数据/代码/配置 SHA。

独立协议审计发现，原通用 prompt 允许 action 或 verdict，但动作阶段解析器只接受 action；这会使 Zero-shot 因遵循通用指令而被拒绝，无法公平比较。正式闭环统一在原 protocol 后、公共 JSON payload 前，为 Zero-shot Active 与 Action SFT Active 添加同一条明确的 acquisition-only 指令：只返回下一合法观察 action，不返回 verdict 或提前停止，由独立冻结诊断器完成最终判断。指令原文及 SHA 记录在运行配置中。该修复由独立协议审计驱动，属于接口指令与解析规则对齐；先前失败/未完成运行完整保留并排除正式比较，不按成绩选运行。原训练分布、checkpoint、监督 prefix 学习评测、token 上限和最终诊断 prompt 均不改变，不重训或按 Validation 调参。

环境在模型推理前私下按 raw 腐蚀记录实例化候选；所有策略推理完成后才载入私有 annotations/曝光表评分。模型输入和传输接口从不接收 Gold、Oracle、私有几何或未来观察。原 Fixed/Random planner 使用环境的关节限位来去重和恢复状态，保留这一原控制实现；不能将其描述为与模型完全相同的信息访问。报告合法动作率、证据获取率、诊断精确准确率、观察成本及每物体配对差异。证据率同时报告全案例分母和缺陷案例分母，避免 clean 案例被当作取证失败造成误读。曝光为 counterfactual pixel proxy，不是语义识别。私有 Oracle 仅为原有限网格特权参考，非全局最优证明。

问题一由训练 loss 与固定监督 prefix 上的动作输出/概率检验回答；不能用 legality 或首步 front 学会了替代动作学习。问题二由真实闭环证据改善回答，诊断准确率另列。若无收益，区分可见信息不可识别、动作监督未学会、分布偏移、取证预算饱和及诊断器瓶颈；保留负结果，不增数据、改预算或按 Validation 重训。

## 数据复现

从 SFT 仓库根目录运行（输出目录必须不存在；不得覆盖本轮产物）：

```bash
PYTHONPATH=. python scripts/export_action_distributions.py \
  --source ../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4 \
  --out data/partnet-action-distributions-v1 --splits train validation
PYTHONPATH=. python docs/evidence/partnet-action-sft-v1/audit_data.py
```

训练、checkpoint 与闭环复现命令及最终实际结果由本轮总报告给出。此文档仅确定协议和训练前审计，不宣称训练或闭环已经完成。
