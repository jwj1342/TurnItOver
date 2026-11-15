# Vulcan 截稿实验结果（2026-11-15）

本目录是在 Vulcan 上补跑与汇总的结果，供填写论文表格。论文正文没有改动；这里给出数字、对应的 LaTeX 行和每个数的出处。代码基线是分支 `exp/vulcan-900`，数据是 `three-tier-900-sft-v1`（manifest sha256 `6d0fc534…`）。

## 运行环境

- 识别：Qwen3-VL-2B/4B/8B 加 LoRA（r8，α16）。17 个已有 checkpoint 在 Nibi H100 MIG 上训练并评测。8B Ordinary 的第 3 个种子（20261213）在 Vulcan L40S 上训练。测试集评测、withheld、各观测条件评测和验证集同硬件重评都在 L40S 上完成。
- 修复：生成器是开源的 **Qwen3.8-27B-FP8**（revision `c3a462cc`），用 vLLM 0.32.0 在 L40S 上单卡起 6 个副本。thinking 开启，`reasoning_effort` 用默认的 xhigh，采样参数沿用模型自带的 temperature 1.0、top_p 0.95、top_k 20，max_tokens 为 32768。Kimi K3 中转站从 11-15 14:40 起一直返回 503，所以 frontier 生成器那一列还没有数。
- 每个结果文件都写明了它用到的运行目录。原始输出在 `tio-runs/out/` 和 `tio-runs/repair/q38/` 下，不进 git。

## 表格与文件对应

| 论文表 | 文件 | 说明 |
|---|---|---|
| `tables/recognition.tex`、`recognition_sizes.tex`（验证集） | `recognition-validation.md` | 现有数字逐位复现；8B Ordinary 补齐 3 个种子 |
| 同上，测试集（新增） | `recognition-test.md` | 180 例，全部在 L40S 上 |
| 跨硬件对照 | `recognition-validation-l40s.md` | 全部验证集评测在 L40S 上重跑一遍 |
| withheld-evidence 段落 | `withheld.md`、`withheld-l40s.md` | 47 例；后者的 full 列和 withheld 列在同一硬件上 |
| `tables/acquisition.tex` | `acquisition-rows.md`，依据 `exposure-conditions-*.md` 和 `condition-f1.md` | 900 版：Exposure 按 case 计；Defect F1 对全部标注缺陷计分 |
| `tables/oracle_selection.tex` | `budget-selection-test.md` | 代理结果，限制见下文 |
| `tables/repair.tex`、`repair_breakdown.tex`、`repair_cost.tex` | `repair-*.md`（跑完后补） | 资格集 58 例，见 `repair-eligible.txt` |

## 解读这些数字前必须知道的问题

下面这些问题由零信任代码审查发现，并逐条在代码里核实过。它们不改变已测出的数字，但会改变数字能支持哪些说法。

1. **Ordinary 的证据支持度为 0，是由定义直接推出的。** Ordinary 的训练目标引用全部观测步（`code900/turnitover/oracle/sft_data.py`），支持度却要求引用的步骤是 gold 步骤的子集（`sft_metrics.py`），而 gold 从不引用全部步骤。所以“0 vs 33.4”不能当作实验发现。
2. **双缺陷样例只给第 1 个标签记暴露。** `release_sft_export.py` 把所有像素差都记在 label 0 上，Hard 档的第二个缺陷永远算“未暴露”，`part_rotation` 从来没被计分，All F1 的上限是 80。
3. **两种监督方式在 600 条训练样本里只有 45 条的标签不同。** 差别主要在引用步骤的格式上。2B Evidence 在 Easy 上不同种子从 14.1 到 43.5。8B 补齐 3 个种子后，验证集 All F1 是 Ordinary 23.0、Evidence 18.9，测试集是 24.9 对 23.3。只有 2B 两个划分上都是 Evidence 更好：验证集 24.6 对 8.9，测试集 23.9 对 11.9。
4. **零样本得 0 分，主要是格式问题。** 模型照抄了提示词里唯一的空 uncertain 示例，不能说明它看不出缺陷。
5. **没有训练目标是 pass，clean 候选与参考图逐字节相同。** 因此状态准确率和 false pass 靠像素差分就能拿满分，不能当作模型理解的证据。
6. **硬件会影响结果。** 同一个 checkpoint，在 L40S 和 H100 上评测会差几个点：2B Evidence 验证集 All F1 是 22.8 对 24.6。主表里 8B Ordinary 的第 3 个种子来自 L40S，需要注明。
7. **withheld 的 47 例有选择偏差。** 被剔除的 13 例里，11 例所有格子都暴露缺陷（其中 9 例是 `part_offset`）。留下的格子全是像素完全相同的，所以 pixel judge 必然得 100%。
8. **`oracle_selection` 是代理结果。** 论文说的是“单独冻结、动作网格经过审计的留出集”，Vulcan 上没有这份数据。这里的结果来自 900 测试集的像素暴露记录：按 case 计的指标可靠，按标签计的召回只用单缺陷例子。动作网格只有 2 到 6 格，所以 K≥2 就基本饱和了。
9. **论文正文和这套实验不一致。** 3、4 两节写的仍然是 benchmark/v1、OpenHands、两个生成器、零样本 4B judge、Claude judge、r4/α8，而 `tables/datasets.tex` 和这里的结果都是 three-tier-900、r8/α16、自建修复循环和 8B 零样本 judge。

## 修复循环在 Vulcan 上做的修正

开跑前在 `exp/vulcan-900` 上合入了以下修正，所有修复结果都基于修正后的代码：

- **生成器提示词省略几何数组。** 数值数组换成 `__GEOM_k__` 占位符，并禁止编辑。58 例里候选和参考的数组逐例相同，缺陷都在可见文本中。程序文本从 90 M 字符降到 0.25 M。
- **judge 的取景与训练图一致。** 9 例 66 张图中，52 张逐字节相同，其余只差 1 到 3 个边缘像素。
- **每种 judge 条件给生成器的图完全相同。** 都是每个观测格的“参考图 + 候选图”，各条件只在 verdict 文本上有区别。
- **judge 提示词的 part 顺序与导出一致。** pixel judge 的阈值改为在训练集上拟合的 0.0010。
- **生成器失败只结束当前一例**，记为 `generator_error`，重交时补跑；截断的回复不写缓存；作业退出码如实反映失败。
- **汇总以资格集为分母**，缺失和出错的例子算失败；Success 附 Wilson 95% 置信区间；Resolved 取最后一次被采纳的补丁；按 taxonomy 的观测要求合并出“需要运动 / 静止可见”两列。
