# 审计收尾：WARN / provisional

2026-10-17。技能：experiment-audit。独立审查员：fresh GPT-6-Astra ultra，same-family；执行者补充：primary agent。最终独立汇总签字不可用：审查员在汇总阶段返回 usage limit。此文件是执行者对已交付审查消息与实际产物的整理，不冒充审查员最终报告。

原始独立初审保存在 `independent-initial-audit.md/json`：其 FAIL 指向旧冻结证据评分存在不可能通过的 offset 案例，且当时新训练尚未完成。保持初审原文，不将其状态改成 PASS。

## 独立审查员随后实际完成的核验

- 新两组输入108 Train/30 Validation行完全配对，图像、公开提示与诊断target相同；中间组只增加comparison。552图像引用/每组、458唯一PNG/每组及1656源tile逐像素核对通过；失败的初始`*-data`被排除。
- 两组各216连续、有限loss训练步、108唯一训练ID；checkpoint SHA及train SHA与run记录一致。
- 两组transfer各32行，提示/图像/诊断target配对；72唯一面板、288原始tile一致。
- 独立检查原始冻结语义标签、pixel_signals曝光及修正评分规则；24个B/C案例都存在合法可通过的答案，非法/重复/字符串step被拒绝。
- 最终10个评分目录使用同一scorer SHA。审查员用独立实现复核全部536行的分类、证据、完整正确，全部一致。
- 冻结B两组分类及完整正确均4/12，offset均0/4；中间监督不构成缺陷诊断改善。主验证仍是可见性代理，不能提升成三维语义结论。
- 去图像后，直接组分类标签30/30不变，完整四字段27/30不变；中间组二者均25/30不变。

审查消息引用记录在同目录 `reviewer-trace.json`。上述属于已交付核验，未取得后续的独立最终签字。

## 执行者完成的收尾核验

原始冻结inputs/Gold与预存manifest哈希一致；未修改冻结Test。两组本轮checkpoint再次验SHA。14项针对性单元检查通过，Git diff检查通过。

审查员报告过一次复现脚本引号错误；当前归档版本语法检查通过。执行者按文档运行prepare-only，616行派生JSON完全一致，全部图像哈希匹配；结果见`reproduce-verified.json`，未重复GPU训练。原始生成builder的三份快照经编辑差异恢复，字节SHA逐一匹配生成manifest。

最终scorer按组提供正确分母，未采用原始推理metrics中带前缀condition造成的full_motion_n=0。通用evaluator已修复未来计数；已有原始推理文件不改写，最终统计以final-scores为准。派生transfer继承的rows:146过时字段保留，并单独说明实际32行；未来builder已去掉该字段。

## 结论边界

评价类型是simulation_only；中间比较与部分可见性标签是RGB代理。无预测生成Gold或按模型自身数值归一化。训练未见对象与Train隔离，但冻结B/C复用了已开发验证物体；总7个验证物体、1 seed，额外2物体没有axis/offset阳性。joint已在输入提供，不是开放定位。unsupported_finding包含类型、定位和证据错误。

完整性结论WARN/provisional，旧证据评分切片FAIL已通过新离线scorer修正。支持“本轮未显示可靠诊断改善，应停止重复2B LoRA”；不支持“中间监督学会了三维运动语义”或“已解决offset”。
