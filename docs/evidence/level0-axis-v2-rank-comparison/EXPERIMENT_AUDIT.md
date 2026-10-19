# 实验完整性审计

完成日期：2026-10-19。审计者：独立新上下文 GPT-6-Astra ultra，`review_independence=same-family`、`acceptance_status=provisional`。首次审计因额度中断，用户续行后在同一审计代理完成；未重跑训练或 GPU 评测。

总体结论：**WARN：数值和数据完整性通过；范围和实际训练模式需要限定。** 未发现伪造 GT、按模型输出自归一化、虚构结果或隐藏标签进入模型输入。

| 检查 | 状态 | 独立核查证据 |
| --- | --- | --- |
| A. GT 来源 | PASS | `turnitover/oracle/level0_axis.py:24` 从已审核 verdict 取标签；`level0_axis_v2.py:35` 核对 56 例审核/Gold/targets。413 个导出文件与指定提交 Git blob 相符，412 个 manifest SHA 相符。 |
| B. 指标分母 | PASS | `scripts/evaluate_level0_axis.py:41` 按全部样本的标签一致性计分；`evaluate_level0_pixel_baseline.py:27` 仅用 Train 拟合阈值，没有预测统计量归一化。56 个像素分数独立重算成功，clean 分数全部为 0。 |
| C. 真实产物与声明 | PASS（模式已更正） | 所有预测、metrics、summary、132 步 loss、实际 adapter 大小及 SHA 均匹配。两组顺序及共同配置相同；远端 11 个底座文件和 8 个部署代码 SHA 匹配。 |
| D. 实际执行路径 | WARN | 计分与像素 score 确实调用；trainer 启用 checkpoint 标志但未显式调用 `model.train()`。底层 Qwen 为 eval，重计算路径未生效。报告已明确记录；autograd/optimizer 仍实际运行。 |
| E. 结论范围 | WARN | 只有一个 seed、6 个独立留出物体；配对样本相关。一例差异不支持稳定 rank 排序；像素基线更好，不支持轴语义、主动取证或真实场景泛化。 |
| F. 评测分类 | simulation_only | 人工审核并与构造 Gold 一致的渲染图像标签；不是模型生成 reference 的 proxy，也不是人工直接评价本次模型回答。 |

独立重算：Base、rank4、rank8、pixel 的 Train 总体分别为 22/44、44/44、44/44、44/44，Held-out 为 6/12、11/12、10/12、12/12；clean/axis 分项、错误 ID 均与 [REPORT.md](REPORT.md) 和 [summary.json](summary.json) 一致。

运行模式在实际远端环境通过 CPU/meta 探针检查，详见 [runtime-mode-audit.json](runtime-mode-audit.json)，包含完整复现命令。两组 `Qwen.training=False`，54 个 checkpoint 标志为真的模块中，满足 training 条件的为 0；attention dropout=0，无非零 Dropout 模块。两组模式相同，不需要为修正文档重跑已观察的留出集。成本合计 53.049 秒是占用 GPU 的进程墙钟估算，不是 GPU 内核忙碌时间。

结论影响：当前协议下 rank4 有限泛化收益、rank8 未显示额外收益以及下一轮优先 rank4 的建议均获支持；稳定优劣、广泛泛化和语义能力结论不获支持。保留已完成运行及上述限定，没有待修复的数据或数值阻断项。
