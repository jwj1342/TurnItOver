# Oracle 证据与训练导出

本阶段使用 `replicacad-v1` 的 96 个样本，保留原来的资产组划分。目标是区分
“动作没有取得有效证据”和“模型取得证据后仍未识别”，尚未训练视觉模型。

## 自动证据标注

`turnitover/oracle/evidence.py` 从正确资产重放腐蚀，并核对程序 SHA。
对每个有限观测状态渲染候选、正确参考以及逐项撤销腐蚀的反事实。
候选同时相对正确参考和相对反事实达到像素差分阈值，才为该项缺陷记一次暴露。
默认归一化像素 L1 ≥ 0.001、变化像素占比 ≥ 0.001、单像素通道差异阈值 0.02。
这只是可见差异代理，并不能证明人类或视觉模型能识别缺陷类别。

观测网格包含配置的全部静态视角，以及 front、oblique_fr 下各关节的非零离散状态。
预算内的连续轨迹只使用正式观察动作；恢复关节和切换视角都收费。
负下限、零上限关节通过 `limit` 恢复零位，不能假定 `zero` 一定是零位。
每一步实际连续执行的图像必须与独立渲染的目标状态逐像素相同。
离线教师为建表调用的渲染不计入模型预算，另行记录调用数。

Oracle 按整段获取路径的新增暴露数/动作成本贪心选择，并非全局最优策略。
固定对照按视角访问静止状态及该视角的各关节状态；随机对照以样本种子打乱观测目标。
两者是有限观测目标顺序对照，不能等同于任意原子动作空间中的最优基线。

## 已验证结果

`data/benchmarks/oracle-v1/manifest.json`：96 个样本、120 项缺陷、6 步预算；
6,432 次教师建表渲染，三种策略全部实际重放成功。

| 动作预算 | Oracle 暴露数 | 固定顺序 | 随机顺序 |
| --- | ---: | ---: | ---: |
| 1 | 32 | 32 | 32 |
| 2 | 97 | 57 | 43 |
| 4 | 115 | 57 | 53 |
| 6 | 115 | 76 | 66 |

Oracle 未暴露的 5 项均为部件偏移；它们在整个有限网格内都未达到阈值，
不是本次六步预算不足。684 项关节轴反事实静止状态检查的像素差分全部为零，
符合轴错误不改变静止状态的预期。这些统计是全数据工程诊断，不是测试集模型成绩。

## 两种独立训练条件

- `oracle-v1/{train,validation,test}.actions.jsonl`：384/96/96 条动作 BC。
  每条只含单张静态参考和此前已执行的候选观察，下一步图像和真值不进入 messages。
  `annotations.private.jsonl` 保存教师真值、差分信号、轨迹与暴露表，不能作为模型输入。
  当前只监督获取动作，没有单张参考条件下的停止/最终诊断监督。
- `paired-v1/{train,validation,test}.diagnosis.jsonl`：192/48/48 条成对参考诊断，
  三种策略各一个最终诊断。输入额外获得每步同相机、同关节状态的正确参考，
  与动作 BC 的单参考条件不同。仅输出已暴露的缺陷；无暴露时为 uncertain。
  181 条 fail、107 条 uncertain，没有 pass 监督。严重程度来自注入真值，置信度未经校准。

`turnitover/oracle/training.py` 将导出记录适配为多模态处理器输入，并屏蔽全部用户/图像 token，
只监督 assistant 答案。已使用本地 Qwen3-VL-2B-Instruct 处理器核对跨三个 split 的
六条试跑记录（1 张与 6 张图像两种长度），模板前缀一致且答案监督非空。
这是处理器验证，不是训练完成或模型能力验证。

按实际图像 SHA 和可见上下文去掉路径/样本身份后，发现 5 组相同可见前缀对应不同教师动作，
涉及 36 条记录，见 `output/oracle-observable-ambiguity-v1.json`。
因此当前 BC 是特权教师轨迹导出，不能假设教师动作从模型输入中唯一可推断。
后续需处理同观测下的动作分布或建立仅使用可见信息的教师；不能用逐条教师动作准确率
证明动态策略的可学习性。Oracle 暴露率也不能直接视为可达到的模型策略表现。

`scripts/export_action_distributions.py` 已将这些前缀在每个 split 内独立聚合。
`oracle-action-distributions-v1` 中 train/validation/test 分别有 266/64/73 个可见输入组、
275/65/73 条带权目标。权重是组内教师动作的经验频率，并非最优动作概率。
`load_action_groups` 保持同组变体一起加载并检查权重和为 1；训练时应整组打包，
用 `weighted_assistant_loss` 计算答案 token 的加权损失，不能丢弃 `loss_weight`。
单例前缀仍可能包含不可从观测推断的特权目标，该导出没有解决所有可学习性问题。

这也修正了 RP 第 143 行关于“可直接由真值 oracle 序列做行为克隆”的可行性假设：
当前证据证明了特权动作选择的价值，尚未证明这些动作可由单参考观测推断。
RP 原文件为只读，本次实施校核记录于本文。

## 复现

输出目录必须不存在；96 样本任务通过 Slurm 执行。

```bash
source scripts/setup_local.sh
# 六资产烟雾测试
python scripts/run_oracle_supervision.py --out data/benchmarks/oracle-smoke --per-asset 1
# 完整导出
sbatch --account=def-zhouyang scripts/slurm/run_portable.sbatch env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_oracle_supervision.py --out data/benchmarks/oracle-new
# 上一步完成后导出成对诊断
sbatch --account=def-zhouyang scripts/slurm/run_portable.sbatch env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/export_paired_diagnosis.py --oracle data/benchmarks/oracle-new --out data/benchmarks/paired-new
# 已安装的 Qwen 环境；只加载处理器，无需 GPU
HF_HUB_OFFLINE=1 .venv-qwen/bin/python scripts/validate_training_export.py --dataset data/benchmarks/oracle-new --out output/tokenizer-new.json
```

`scripts/evaluate_oracle_qwen.py` 使用冻结的本地模型，对相同轨迹比较单参考和成对参考。
完整测试是 16 个对象样本 × 3 种获取轨迹 × 2 种参考条件，共 96 次调用。
它只接收公开 prompt/图像，模型返回后才使用完整真值计分；无效 JSON 计漏检并单独报告。
该评估仍不包含模型主动选动作、微调、生成器修复闭环，且测试资产族仅为门。
