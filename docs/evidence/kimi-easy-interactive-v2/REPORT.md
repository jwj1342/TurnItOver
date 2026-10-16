# Kimi Easy Object 完整交互教师实验（2026-10-16）

## 1. Kimi 能否完成自主取证与正确诊断？

部分能，但本轮尚不能稳定提供三类可靠诊断。18 个保守筛选的高置信度案例均执行了一次
完整采集尝试；15/18 得到合法终判，10/18 严格正确（55.6%）。本实验的交互范围是
正常资产预先选定的关节和原人审视角，不是全视角搜索或多关节自主定位。

| Gold 类型 | 案例 | 合法终判 | 合法终判的四字段匹配 | 严格正确 |
|---|---:|---:|---:|---:|
| clean | 7 | 7 | 7 | 7/7 |
| joint_axis | 6 | 4 | 4 | 3/6 |
| part_offset | 5 | 4 | 0 | 0/5 |
| 合计 | 18 | 15 | 11 | 10/18 |

严格正确要求合法 schema、verdict/type/joint/parts 全部匹配、实际图像与参考状态绑定、
finding 的每个引用符合现有独立证据规则；clean 必须覆盖声明范围的全部不同参考状态。
偏移的 joint_id 期望 null；不把关联关节元数据当作错误关节标签。
单独解析原始终判内容时，轴字段为 5/6 匹配，另 1 例连接/超时无返回；其中洗碗机
虽内容匹配但 schema 不合法，所以不计入严格正确或 SFT 诊断。

Kimi 自主执行 54 个观察动作，全部合法，并在 18/18 例中覆盖 rest/mid/limit 的不同
参考状态。初始化仅提供公共 hierarchy（固定 step 0）和正常参考图，不预给候选图像。
17 例自行选择 request_view→mid→limit，1 例选择 actuate(rest)→mid→limit。
所有案例都在三次自主图像观察后请求终判；没有证明相对固定脚本的采样优势。
动作、detent 和结束决定来自真实模型调用，候选结果来自实际 Chromium，不是 FrozenBoardSession。

8 个未严格通过案例的互斥主因如下，详见 [逐案例结果](CASE_RESULTS.md)：

| 主因 | 数量 | 案例 |
|---|---:|---|
| 偏移被明确判 pass | 4 | PM_48686、PM_102016、PM_10697、PM_19836 的 offset |
| 轴类型与定位正确，但引用了无异常 rest | 1 | PM_102016 axis，引用 [1,2,3]，现有规则只认可异常动态帧 |
| 终判缺少必填 limitations | 1 | PM_12594 axis，原始字段与 [2,3] 引用匹配，格式仍被拒绝 |
| 连接/超时，无终判返回 | 1 | PM_19836 axis |
| 4096 输出 tokens 全部用于 reasoning，无终判文本 | 1 | PM_102379 offset |

没有 uncertain 输出，不补造 uncertain。偏移的 4 次 pass 均已观察充分状态，属于对本轮
目标的明确漏判，不能归为动作未执行。Box 的未终判则属于响应失败，不能算作确定漏判。
仅三个轴诊断严格通过：PM_10697、PM_5088、PM_48686。

运行先完成 PM_10697 axis 一例，验证真实循环、审核与后续导出接口，再串行完成另 17 例。
每例只有一个 Episode、4 次实际请求，合计 **72 次请求尝试**；不重试、不换模型、
不改失败案例 Prompt。上限预设为每例 8 次、总计 144 次，实际没有触顶。
请求模型及成功响应报告的模型均为 `k3-256k`。

已知用量为 input **182,609**、output **78,083**，合计 **260,692 tokens**。
其中 74,226 reasoning tokens 已包含在 output 内，不重复相加。
一次空文本错误的原始 metadata 内仍有 7,014 tokens usage，已恢复计入；一次连接/超时
没有 usage，所以以上只是已知小计，实际总量未知。billing_cost_rmb 保持 null。
原始 Episode/metadata 不因补账而改写，另存每例 token_accounting.json 和 api_calls.jsonl。

## 2. 实际获得了多少可信 SFT 数据？

| 导出类型 | 条数 | 独立资产 | 限定含义 |
|---|---:|---:|---|
| 动作监督 | 52 | 7 | 合法、到达新状态、实际图像绑定冻结参考；不表示最优动作 |
| 诊断监督 | 10 | 7 | 7 clean + 3 joint_axis；0 part_offset；0 uncertain |
| 合并 Episode 行 | 62 | 7 | 上述两类的合集，不能再与两类条数相加 |

原始 54 个自主动作中，Box 的两个 mid 与 rest 为相同实际状态，没有新增信息，故
拒绝这 2 条动作监督。动作按独立的状态新颖性和参考可比性审核，不按隐藏缺陷标签
选择；错误或缺失终判的 Episode 中仍可保留合格观察动作，但其终判绝不作为正确诊断。
正例缺陷诊断只有 3 个独立资产；“诊断覆盖 7 个资产”包含全部 7 个 clean 对照。
偏移五例没有产生一条合格诊断，因此当前数据不完整覆盖三类目标。

数据在 [sft/manifest.json](sft/manifest.json)，动作/诊断分别存入
`sft/train.actions.jsonl` 和 `sft/train.diagnosis.jsonl`。
拒绝原因在 `sft/rejected.jsonl`，合理 uncertain 独立文件 `uncertain.jsonl` 本轮为空。
没有合并旧 117 条，也没有训练。所有源资产为 Train；validation 文件为空，不涉及冻结 Test。

真实 SFT loader 已读取 52 actions、10 diagnosis、62 combined rows；全部图像 SHA、
来源身份和正式 Train 数据属性检查通过。Qwen3-VL-2B 本地 processor 处理了全部
**62/62 行，失败 0**；assistant-only loss mask 全部通过，最长输入 3,082 tokens。
额外逐行核实：user Prompt 与实际教师调用完全一致，assistant JSON 内容与模型决策一致，
图像顺序和 SHA 与每次模型实际收到的图像一致，没有未来观察或 Gold 加入学生输入。
证据见 [processor_validation.json](processor_validation.json)、
[sft/loader_validation.json](sft/loader_validation.json)、teacher_image_manifest.jsonl。

没有发现教师—学生输入错配，但有重要的语义协议限制：任务范围称为“单关节运动”，
未同等明确强调静态安装对齐；一些漏判解释只检查运动机制。轴的无异常 rest 可以
作为基线语境，但现有 finding 审核要求每条引用都显示差异，教师一次没有遵守。
所以偏移结果不能单独用于证明 Kimi 纯视觉能力不足；引用不合格也不等于类型识别错误。
这些限制与原文均保留于 [PROTOCOL_LIMITS.md](PROTOCOL_LIMITS.md)，没有事后改 Prompt 或修剪引用来提分。

同样需要限定：公开 task/reference 已告知正常选定 joint 和 moving part，定位计数
不证明模型能从任意多关节物体中自主发现缺陷关节。本文的可靠监督仅对实际保存的
受控输入协议成立，不外推为一般交互能力或已验证的训练收益。

复现命令见 [REPRODUCE.md](REPRODUCE.md)。`config.json` 冻结 cohort、源代码和 Prompt 哈希；
`prepared.jsonl` 保存实际公开任务和范围；private/frozen 保存 Chromium 参考/候选及 checker；
cases 保存每次原始 Prompt、响应或错误、因果轨迹、图像和审核。
`artifact_manifest.json` 对本实验文件逐一记录 SHA-256。
相关单元测试 **51 项通过**，真实 Chromium 测试 **1 项通过**，没有声称整库测试或训练已完成。

## 3. 下一步应该做什么？

选择：**优先修复教师交互或审核问题**。

关键证据是：18 例均已取得范围内完整状态，但仅 10 例严格通过；5 个偏移案例没有
可信诊断监督，另有轴引用、缺字段、输出截断和连接/超时失败。此时继续补对象不能
直接解决已观察到的协议/终判可靠性问题，而直接 SFT 会缺少偏移正例，动作数据也
主要是同一状态扫描。loader 能读取不等于监督目标已经完整可靠。
当前保留这批 52 条动作、10 条诊断作为独立、可复核的候选数据，不正式启动本轮 SFT。
