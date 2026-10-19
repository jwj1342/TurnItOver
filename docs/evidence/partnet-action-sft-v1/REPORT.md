# PartNet Action-only SFT：能拟合动作监督，紧预算取证有局部收益

本轮实际完成一次 Qwen3-VL-2B-Instruct LoRA Action-only GPU 训练、checkpoint 保存/重新加载，以及真实 Chromium 的四策略闭环。**动作监督可以被小模型拟合；相对 Fixed 的取证收益只出现在 K=2/4，且全部集中于一个 Validation 物体。K=6 反而不如 Fixed，不能声称学得稳定更优的主动策略。**

用户已授权改用现有 PartNet，ReplicaCAD 暂不处理。保留开始时未推送的 `b752847` 和入口提交 `78a0c78`；不重训 v4/v5、不扩数据、不根据成绩选 checkpoint，不访问 frozen Test。目录名 `oracle-first-failure-v4` 是本次复用的旧监督版本，不是本轮诊断训练。

## 数据和一次训练

来源为 dataset 工作树现存的 `data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4`。Train 28 个物体/110 案例/660 条旧动作，Validation 7 个物体/28 案例；其中 19 个缺陷案例含 **29 项注入缺陷**，另有 9 个 clean 案例。物体、来源身份/内容和参考图像没有跨 Train/Validation 重叠，828 个引用图像路径的声明 SHA 全部匹配。模型只接收当前静态参考、已取得观察、公开上下文和剩余预算。

复用既有 exact-prefix 聚合：Train 496 组/500 个动作变体，Validation 121 组/122 变体；Train 3 个冲突组，Validation 1 个。保留全部组内经验概率，不能将任何一个动作当作唯一 Gold。这些概率来自特权教师频率，不是可见信息下的最优策略。分组、身份、图像 SHA 及预算预审计见 [data-audit.json](data-audit.json) 和 [PROTOCOL.md](PROTOCOL.md)。

从原始底座独立初始化：rank=4、alpha=8、LR=2e-4、seed=20261017、3 epochs；assistant-only token loss，组内概率加权、组间等权，每组更新一次。实际 **1488 updates**，1,605,632 个可训练参数；347.918 秒，峰值 torch allocated 10.724 GiB。三个 epoch 的在线平均 loss 为 `0.062746 → 0.015964 → 0.014310`。这是不同参数状态下的训练日志，不能当作最终 checkpoint 的统一 NLL。

最终 adapter SHA-256：`0fc95434c665b4ee587c5d8cf930408d28e3c6cb909057c08e7eeae21745d346`。基座权重 SHA、所有训练输入 SHA、配置与逐更新 loss 分别保存在 [train-run.json](train-run.json)、[train-loss.jsonl](train-loss.jsonl)。训练与模型推理按同一远端环境和底座路径启动；adapter 重新加载日志明确记录了相同 SHA。base-only 服务的旧 ready 日志仅有 host/port，未保存运行时底座权重/环境/代码 SHA，不能声称已独立核验该服务当时加载的底座字节；当前复现脚本不能追补为历史运行时证明。

## 问题一：能否学会现有动作监督？

在最终 checkpoint 上重新评估所有原始 Train/Validation 公共前缀，未按结果修改模型。NLL 为组内经验概率加权的 assistant-token 平均，再对组取平均。预测“经验质量”指生成动作在原分布中的概率质量，不是动作的真实最优概率。

| split | 底座 NLL | Action SFT NLL | SFT 合法率 | SFT 支持动作率 | SFT 经验质量 |
|---|---:|---:|---:|---:|---:|
| Train，496 组 | 1.473898 | 0.009021 | 100% | 93.55% | 93.30% |
| Validation，121 组 | 1.460329 | 0.045325 | 100% | 84.30% | 83.88% |

剔除统一首动作 front，Validation 剩余 114 组的支持动作率仍为 **83.33%**、经验质量 82.89%、NLL 0.048108。因而结果不只是在学首步 front；Train/Validation 差距也表明尚未完全泛化。完整记录见 [learning/metrics.json](learning/metrics.json)、[learning/groups.jsonl](learning/groups.jsonl) 和 [learning/stages.json](learning/stages.json)。

原始监督提示词允许 action 或 verdict，底座在该提示下自由生成主要返回 verdict；因此原前缀上的底座低合法率不能解释为动作选择能力低。NLL 下降与动作匹配支持“拟合了经验监督”，但不能证明视觉理解或最优主动决策，也没有图像消融来证明图像依赖。可见输入冲突确实存在，单例也未必可辨；但当前绝大部分目标已能拟合，不能把所有剩余差距都归因于不可识别。

## 问题二：闭环是否改善取证？

同一 7 个 Validation 物体、28 案例、参考 PNG 原字节、允许动作、状态规则和 K=2/4/6，实际完成 **336 条闭环、1344 次收费观察**。复用原 Loop、动作执行器、Fixed/Random witness planner。Oracle 只在结束后用原曝光表和原 planner 计算离线特权参考。Fixed/Random 使用环境关节限位生成恢复动作，是保留的结构化 planner 控制；不能称为与模型完全相同的信息访问。

四策略的合法动作率均为 **100%（336/336 次动作/策略）**；每案例成本均为对应的 K，没有超预算或模型查询 Gold。两种 Active 共用同一 acquisition-only 指令。环境私下实例化候选；模型请求只含公共 prompt/images，私有标签、反事实和未来观察不传入模型。所有策略最终诊断均使用同一底座并关闭 Action adapter。

下表“证据”是后验反事实像素曝光代理，分母为 29 项注入缺陷；不是语义识别率。“有证据案例”只在 19 个缺陷案例上计数。

| K | 策略 | 证据曝光/29 | 有证据案例/19 | 全部缺陷已曝光案例/19 | 最终诊断正确/28 | 最终输出有效/28 |
|---|---|---:|---:|---:|---:|---:|
| 2 | Zero-shot | 10 | 10 | 2 | 0 | 10 |
| 2 | Action SFT | **17** | **14** | **9** | 1 | 15 |
| 2 | Fixed | 15 | 13 | 6 | 2 | 8 |
| 2 | Random | 13 | 12 | 4 | 1 | 10 |
| 4 | Zero-shot | 10 | 10 | 2 | 1 | 13 |
| 4 | Action SFT | **19** | **15** | **10** | 4 | 16 |
| 4 | Fixed | 16 | 14 | 7 | 5 | 21 |
| 4 | Random | 17 | 14 | 8 | 3 | 11 |
| 6 | Zero-shot | 10 | 10 | 2 | 4 | 17 |
| 6 | Action SFT | 22 | 16 | 13 | **9** | **25** |
| 6 | Fixed | **25** | **17** | **15** | 2 | 9 |
| 6 | Random | 19 | 16 | 9 | 6 | 17 |

离线特权 greedy Oracle 的曝光为 K=2：27/29，K=4/6：29/29；它不是可由公共输入达到的承诺，也不是全局最优证明。它的模拟渲染/私有表不进入模型，不计作 learner 成绩。

相对 Fixed，Action SFT 的曝光率差分别是 **+6.90、+10.34、−10.34 个百分点**；相对 Random 为 +13.79、+6.90、+10.34；相对 Zero-shot 为 +24.14、+31.03、+41.38。所有方法的成本相同。Zero-shot 在 K=6 的 168 个动作全部是 front，140 张观察与此前取得图像重复；SFT 的重复图像为 3/168。它确实执行了更多不同取证动作，但不能仅由此推出视觉策略语义。

逐物体曝光增量如下；每格依次是 K=2 / K=4 / K=6，统计该物体全部案例取得的缺陷项数差：

| 物体 | SFT − Fixed | SFT − Random | SFT − Zero-shot |
|---|---|---|---|
| PM_100202 | 0 / 0 / −1 | −1 / −1 / −1 | 0 / 0 / 0 |
| PM_102252 | 0 / 0 / −1 | +1 / +1 / +2 | +1 / +1 / +3 |
| PM_102590 | 0 / 0 / +1 | +1 / +1 / +1 | +1 / +2 / +2 |
| PM_10797 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| PM_1280 | **+2 / +3 / 0** | +1 / 0 / +1 | +3 / +4 / +5 |
| PM_34610 | 0 / 0 / −2 | +1 / 0 / −1 | +1 / +1 / +1 |
| PM_45759 | 0 / 0 / 0 | +1 / +1 / +1 | +1 / +1 / +1 |

完整逐案例/逐物体诊断与成本差保存在 [comparison.json](comparison.json)。紧预算下优于 Fixed 的增益全部来自 PM_1280，不能推广成多物体稳定收益。K=6 时 Fixed 已很强，而 SFT 仍错过其他物体的证据；这既可能涉及长历史策略泛化和经验教师频率，也说明当前任务有强固定路径，不支持简单宣称“不需要主动选择”或“监督完全不可学习”。本轮未做同预算的去历史/只上下文控制，不能区分依赖新图像的主动选择与学到的条件固定动作序列。不扩大数据、不改预算或重训来改善本次结果。

最终诊断准确率只比较 defect-ID 集合与 pass/fail 状态，不覆盖部件/关节定位、解释正确性或证据语义。无效 JSON/字段、非法引用和截断均保留为失败，不从准确率分母剔除；当前冻结诊断器是明显瓶颈：**四策略在每个 K 下的 19 个缺陷案例全部为 0/19 正确**；表中所有正确诊断都来自 clean。K=6 的 SFT 9/28 正是 clean 的 9/9，不能称为恢复了缺陷诊断。旧 final_prompt 的 seven/eight 字段文字矛盾保持冻结，未按本轮结果修改。K=6 SFT 的诊断高于 Fixed，但其曝光反而更少，不能把该诊断差异当作更强取证的证据。

## 协议修复、复现与保存

原始入口评测中发现了两类工程/协议问题：本机代理将 localhost 发送至代理而 502；旧原始提示允许 verdict、动作校验却禁止 verdict。分别停止于 `baselines/`、`baselines-run2/`，保留全部原记录，排除正式比较。独立 fresh-agent 审查要求对两个 Active 加相同的 acquisition-only 阶段指令；不改训练消息、不重训、不改 cap、预算、病例或 frozen final。正式成绩只来自 `baselines-run3/` 与 `action-sft/`。原始前缀学习评估保留旧提示，因此与闭环阶段的合法率不能直接横比。这是记录在案的协议修复，不是一次未经修正的预注册评测。

单 seed、小样本，证据是像素代理；不主张统计稳定排序、视觉语义、跨分布泛化或真实 POMDP 最优性。K=2/4 对相对 K=6 监督还有预算条件分布偏移。本轮的负结果（K6 不如 Fixed、诊断错误和动作匹配不足）全部保留。

- [run-training.sh](run-training.sh)、[run-learning.sh](run-learning.sh)、[run-base-server.sh](run-base-server.sh)、[run-sft-server.sh](run-sft-server.sh)、[run-closed-loop.sh](run-closed-loop.sh) 给出实际参数。输出目录必须不存在；先按 PROTOCOL 导出旧动作概率，部署到同一远端根目录。
- 本机原始动作、观察 PNG、prompt、模型响应、逐 episode 错误、配置、SHA 和checkpoint保留于 `output/partnet-action-sft-v1/`。远端训练/学习工件保留于 `/home/laq/experiments/partnet-action-sft-v1/`；本机 Chromium 工件为权威轨迹。Git 仅提交轻量记录，不提交模型权重或大图像包。
- SSH转发：`ssh -N -L 8893:127.0.0.1:8893 -L 8894:127.0.0.1:8894 laq@10.201.129.199`；分别启动两个 server 脚本后执行闭环脚本。没有付费 API。
- 环境见 [ENVIRONMENT.md](ENVIRONMENT.md)，测试见 [tests.txt](tests.txt)。模型 loss/动作概率、闭环证据和最终诊断始终分开解释。

原始闭环还打包保存为 `output/partnet-action-sft-v1/real-closed-loop.tar.gz`（含全部 prompt/原始输出/PNG/episode/学习评测，不含权重），SHA 见 [bundle.sha256](bundle.sha256)。全工件逐文件 SHA 见 [artifact-manifest.json](artifact-manifest.json)。可无模型调用复核真实步骤与图像、概率目标和计数：

```bash
PYTHONPATH=. ../TurnItOver/.venv-local/bin/python docs/evidence/partnet-action-sft-v1/verify_run.py
```

当前确定性核验通过：336 episodes、672 个 Active 模型动作逐条与实际执行一致、1680 个 PNG SHA、1488 updates，以及真实重载 adapter SHA；见 [verification.json](verification.json)。
