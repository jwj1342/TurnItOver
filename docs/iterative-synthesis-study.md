# 生成—验证—修复闭环：实验方案

## 研究目标

本实验研究主动获取证据的 verifier 是否能改善三维程序合成的迭代效率，而不只提高单轮判定质量。核心假设是：在冻结生成器、初始候选和修复预算一致时，基于主动视角、关节动作与运行时查询形成的结构化反馈，能够提高修复成功率、减少无效修改，并改善停止时机。

`handoff/img2threejs/` 提供了 HTML 原型中的 generate → render gate → active verify → visual revise → fresh verify 控制经验；正式实验实现应复用 TurnItOver 的 Program ABI、浏览器 harness、verifier、deterministic audit 和 repair protocol。交接代码本身不是实验结果。

## 任务与闭环

每个任务包含参考输入、初始生成请求、冻结生成器配置、最大修复轮数和观测预算。执行流程为：

```text
reference input
    → initial program generation
    → executable/render gate
        ├─ failure → bounded runtime repair → gate
        └─ success
    → fresh verification episode
    → structured verdict, diagnosis and selected evidence
        ├─ pass/stop → terminate
        └─ revise → bounded program repair → gate → fresh verification
```

runtime repair 只处理编译、加载或可靠渲染失败；visual repair 只处理通过执行门槛后的语义、几何、结构、运动学和外观问题。任何代码修改后都必须创建新的验证 episode，不复用上一轮相机、关节状态、observations 或 conversation。

## 实验条件

所有条件使用相同任务、初始候选、冻结生成器、生成参数、修复调用预算和确定性私有审计：

| 条件 | 提供给修复器的反馈 |
| --- | --- |
| No-verifier | 不提供验证反馈，执行固定轮次或生成器自行停止 |
| Fixed observation | 固定视角和固定动作序列产生的观察与诊断 |
| Active verifier | verifier 在相同 observation budget 内选择视角、关节动作和 runtime query |
| Oracle observation | 使用真值可检测性选择观察，但诊断仍由待评系统产生 |
| Oracle diagnosis | 使用正常观察，提供由私有真值生成的诊断 |
| Combined oracle | 同时使用 oracle observation 与 oracle diagnosis，作为联合上界 |

私有 deterministic audit 只用于评价候选，不得泄漏给生成器；只有 oracle diagnosis 条件可以接收由真值构造的语义反馈。每个条件应预先固定失败处理、超时、重试和停止规则。

## 指标

主要指标包括最终通过率、首次达到质量阈值的修复轮次、到阈值的模型调用数，以及给定质量下的 token 与 observation cost。

过程指标包括 non-improving iteration rate、质量回退率、exact-state revisit rate、停止误差、不可执行 proposal 比例、render/runtime failure rate 和 quality gain per token。检测与观测还应分别报告分缺陷类型的 precision/recall/F1、定位正确性、缺陷暴露率及预算—性能曲线。

退出码不作为成功指标。每次运行必须记录结构化 termination、最终 verdict、私有 audit、每轮 candidate hash、模型用量和 observation cost。

## 公平性与可复现性

- 按资产族拆分开发集和最终留出集，避免同族泄漏；结果同时报告任务级重复与资产间差异。
- 固定代码 commit、输入资产、非敏感模型标识、prompt 版本、温度、预算、action space、浏览器与 harness 版本。
- 每轮保存输入、允许公开的反馈、候选源码、结构化判定和证据引用；私有 audit 与模型可见数据物理隔离。
- 模型调用失败、candidate 不可执行、预算耗尽和不确定 verdict 都作为正式结果保留，不能只统计成功案例。
- 先用 fake model 和小型浏览器 fixture 验证控制流，再运行付费模型；任何真实模型 pilot 使用新的输出目录。

## 结论边界与阶段门槛

单个 run、少量案例或 Verifier 的 `accept` 不能支持泛化、训练收益或物理正确性结论。单张真实照片也不能提供隐藏结构、绝对尺度与关节轴的完备真值；真实输入评测必须区分可观察外观与不可观察属性。

第一阶段完成标准是正式 outer loop 能在每次修改后重新构造证据，并在所有条件间保持相同预算和审计。第二阶段使用多个资产族和重复采样验证主动反馈的收益。只有在预注册指标上相对 fixed observation 稳定改善，且收益没有被额外 observation/token 成本抵消时，才进入 verifier 训练或更大规模真实照片实验。
