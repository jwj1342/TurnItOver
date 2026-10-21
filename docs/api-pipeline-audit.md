# API Pipeline 完整性检查与审查状态

日期：2026-10-21。**最终新代理语义审查未完成，状态 ERROR，不是 PASS。** 按 [experiment-audit 技能](/home/laq/.codex/skills/experiment-audit/SKILL.md) 启动的 GPT-6-Astra ultra 新代理及其成本检查子代理在产出最终判断前遭额度限制；更早代理也曾网络中断。本文件由执行者汇集已落盘的确定性检查，不冒充审查代理的最终裁决。审查独立性为 same-family，最终 acceptance 为 incomplete。

预检代码审查已单独完成，结论仅限 [预检报告](api-pipeline-code-review.md) 当时的生成入口；不能替代最终实验审查。以下原始检查保存在 [audit-trace](../results/2026-10-21-api-pipeline/audit-trace/)，含代理保存的校验程序、完整数字和输入哈希。

| 检查 | 已核验的事实 | 审查限制 |
| --- | --- | --- |
| A. GT 来源/隔离 | `fixture.py:28–38` 独立作者参考；共享生成源码匹配实际 API 响应且不同于参考；`scoring.py:119–154` 为完成后的独立评分及哈希核对 | 未完成最终逐行语义审查；公开规范本身给出尺寸和运动要求，不能称照片 benchmark |
| B. 分母/计分 | `scoring.py:21–29` 原始 invariant 通过数/总数；没有按模型自身最大值归一化；三初始程序分别32/32、22/32、28/32 | 几何/运行时有限采样，不覆盖材质与全部组合状态 |
| C. 结果/成本 | 原始请求、response/SSE、usage 与九台账独立重算：44次、39完成/5失败、145467已知tokens、4未知usage、货币全未知；`summarize_api_pipeline.py:14–34` 包含失败 | 无外部供应商账单或模型权重/路由证明 |
| D. 实际执行 | 新代理检查脚本实际调用 Chromium 重放全部六组并保存 `arm_replay`；另从归档新解包再次评分，六组质量、诊断、见证与API成本一致 | 脚本 fixture 测试不计真实API效果；图像存在不等于自然语言证据解释正确 |
| E. 范围/恢复 | 单作者物体、三个受控案例、每组一次；c02 原失败历史有哈希，续接请求除stream字段外相同，原始prompt/图像字节相同；仅六个已有汇总/报告文件更新，新调用另存 | c02 分段恢复；Fixed K=4未观察第二关节；不足以支持普遍优势 |
| F. 评价类型 | `simulation_only`：独立作者几何参考、真实 Chromium/Three.js 执行，API生成/诊断/修复 | 非真实照片数据集GT，非人类评价，参考不是模型输出 |

代理保存的 `accounting-checks.json` 中 `problems=[]`；其备注还指出：4096推理耗尽的失败调用在 `metadata.json` 没有填 returned_model/finish_reason，原 `response.json` 保留了这些字段，6393tokens已计入总账。旧记录未覆盖；读取失败结果应使用原始响应，不能只依赖metadata。

可支持的有限结论：真实闭环及一例门轴修复有原始证据，配对输入和成本可核对，报告中的六组数值可离线复算。仍不能宣称审查完全通过、Active普遍优于Fixed、货币成本已知、完整语义证据正确或完整物理正确。最终语义审查中断是本次交付的未解决项。
