# Easy Object + Obvious Defect：第二轮盲审

2026-10-16 更新：24 个候选的人审已完成，生成时的 `pending_human_review` 标记为历史快照。
逐例核对、误判和下一轮候选清单见 [盲审后审核报告](../partnet-mobility-easy-objects-v2-audit/REPORT.md)。
下文描述原始生成与审核流程；自动通过不代表人工通过或训练资格。

打开 `boards/` 的 8 张图，每张依次包含参考和 3 个匿名候选，共 24 个候选。
每格是 Chromium 原生 256×256，列为 rest/mid/limit。请对照
`review_manifest.jsonl` 的合法 hierarchy 和状态值填写 `human_review.jsonl`：
verdict、defect_id、joint_id、parts、evidence_states 与解释。
候选顺序按不透明 ID 排序，不按缺陷类型排列。不要在盲审前打开 `private/`。
人工标注沿用第一轮字段；本轮没有预填答案，不能直接用于训练。

对象选择先于任何缺陷生成，仅依据正常机构和正常运动；10 个正常候选预筛选出 8 个，
排除水壶（细把手遮挡）与折叠椅（细框架交错）。6 个入选对象是单关节，
桌子和门各选择一个大部件，其余关节保持 rest。完整理由和全部视角的正常运动指标
保存于 `private/object_selection.jsonl`，该文件在注入前落盘并记录哈希。
这只是 AI 操作者的 Easy 预筛，不等于人类难度验证。
Train 池共有 28 个对象；第一轮 3 个对象、9 个案例独立保留且没有重复计入。
本轮不是对全部剩余 25 个对象的穷尽视觉审核，不能把 8/25 当作 Easy 通过率。

视角按正常运动的最大侵蚀变化面积确定，随后固定，绝不根据缺陷结果调整。
状态沿用生产 detent：mid 为上下限中点，limit 为上限；对称范围的 Box mid=rest，
Laptop mid 为负值。这些状态不是人为挑选的“闭合/半开/全开”。
每类尝试 3 个固定种子的目标命中注入，轴旋转 60°、偏移 0.18 m；
错目标种子单独记录，不计入目标候选分母；选第一个自动通过尝试，不联合优化视角。
所有尝试的真实匹配渲染指标、checker、拒绝原因和未选中尝试均在 private 中。
part_offset 只改目标 part.origin，保持完整 joint spec；joint_axis 只改一个 axis。

自动门槛只证明几何异常、运行预算和可见变化：轴 rest 不变、动态变化足够，
偏移必须在 rest 有足够连通区域。**不能证明缺陷类型在语义上清楚**。
尤其注意 limit 出框、侧视遮挡、Box 缺少独立 mid、抽屉正面遮挡等问题；
即使自动通过，仍可由人工判为 uncertain 或拒绝。
所有自动通过样本为 `pending_human_review`，人类须逐例确认类型、joint、part 和证据状态。

私有 `gold.jsonl` 是注入真值，绝非人类诊断结论；`checks.jsonl` 是独立几何/运行 checker，
完整程序 spec 单独保存以便重建；`screening.jsonl` 和 `summary.json` 保存指标及计数。
公开图板不含 Gold、注入参数、checker 结论或语义提示。
没有调用 Kimi、训练 SFT、读写冻结 Test 资产；只使用 Train catalog 和 split 元数据。

复现（仓库根目录，使用全新输出目录）：

```bash
python -m scripts.prepare_easy_object_review --config configs/easy_object_review_v2.json --out /tmp/easy-objects-v2-repro
python -m pytest tests/unit tests/browser/test_preview.py -q
```

`manifest.json` 保存配置、catalog、split、视角、生成脚本、全部产物和第一轮文件的 SHA-256。
参考和候选共用正常 framing，未扩大视野掩盖出框问题。像素面积只作工程筛选。
`TEST_RESULTS.txt` 保存验证结果；生成运行本身亦为真实 Chromium 集成测试。

本次汇总：Easy 正常预筛 8/10（80%）；轴错误目标命中尝试 24/24（100%），
部件偏移 21/24（87.5%），3 个偏移尝试因 rest 可见错位不足被拒绝。
入选的 8 个 clean 均由 checker 通过；16 个单缺陷均被几何 checker 拒绝，运行预算通过。
24/24 最终候选仍待人工确认。AI 图像目检备注单独保存在 private，不能当作人审结果。
