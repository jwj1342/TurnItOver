# PartNet-Mobility Level 0 joint_axis v3 扩充报告

生成日期：2026-10-19；人工确认：2026-10-20。状态：**新增20条已获用户人工确认并纳入训练；规模未达到原目标**。本轮只进行数据建设；训练、模型推理、付费 API 均为 0。

| 指标 | 数量 |
|---|---:|
| 已检查来源档案（不同原始 ID） | 160 |
| 结构候选 / 结构拒绝 | 94 / 66 |
| 固定 Validation（保持用途） | 7 |
| 已筛历史独立物体（不当作新容量） | 57 |
| 本轮新独立候选，已全部筛选 | 24 |
| 新增自动通过物体 / 样本 | 10 / 20 |
| 新增自动拒绝物体 | 14 |
| 新增真实人工通过 / 待审核样本 | 20 / 0 |
| 合计自动通过物体 / 样本（含旧数据） | 38 / 76 |
| 最终训练合格物体 / clean-axis 成对样本 | 38 / 76 |

最终训练清单为 **38 物体 / 76 样本**（clean/axis 各38），其中本次新增10物体 / 20样本。当前本地已审计池全部合格候选已纳入，距最低50物体 / 100样本仍差12物体 / 24样本；这不是未下载来源的全局上限。

## 人工确认与训练门禁

用户于2026-10-20明确表示：“我已人工审查，这些都没问题；请把这次更改和上次更改合并在一起提交commit”。该确认覆盖原盲审页的全部新增20条；由代理逐行代录，审核者为user、日期精确到天，未虚构具体时刻。原话、20个case_id、原盲审页SHA及记录人与决策人分别存于 `private/user_review_confirmation.json`；原页面归档于 `private/review_page_before_confirmation.html`。

这是用户报告的人工审查与总体批准，不宣称获得了独立盲审逐例标签或逐字运动意见。pass/fail与缺陷类型按被批准的clean/axis成对设计映射；逐例运动描述由代理查看图板整理并明确归属。`human_review.jsonl` 保存完整记录。只有同物体两条均满足原审核门禁才导出，20条现已全部通过；没有默认批准其他候选。

已执行 `python -m scripts.finalize_level0_axis_v2 --out docs/evidence/partnet-mobility-level0-axis-v3`。`review-expansion.html` 当前显示待审0条；`review_manifest.jsonl` 保留全部76条及实际审核状态。

## 来源、划分与去重

复用本地官方 sapien-sim/PartNetMobility，revision `ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f`，沿用已有非商业研究/教育许可记录及 ShapeNet 条款；未新增许可或再分发授权声明。来源和条款链接、每个物体身份、档案和程序 SHA、几何指纹见 `private/capacity_audit.json` 与 `private/source_catalog/catalog.json`。审计排除固定 Validation、历史 57 物体以及另外 6 个已有正常运动失败记录的来源。新来源只进入 Train 候选池，原有 Train/Validation 映射完整保留。另核对兄弟工作树 UniPhys 正式 v1/v2/v3 的 14 条记录（去重为 5 个原始物体，含 PartNet 6635/6659），与新候选的来源、可用标识、内容和精确几何指纹无重叠；详见 private/protected_pool_audit.json。

与已用 Train、已有 Validation 和冻结 Test 检查原始身份、可用 source model ID、内容 SHA 及原有六位小数顶点集几何指纹。Test 仅作只读身份/几何排重，没有渲染、评估或用途变更；原始 OBJ SHA 与冻结计划一致。Test 元数据缺少可直接对应的 PartNet 源 ZIP 和独立 model ID，因此不能声称跨格式内容 SHA 或缺失 model ID 的全面等价验证；使用可用标识与重算几何指纹补充，亦不声称排除了所有近似重复。

## 固定协议与拒绝原因

沿用原有生成器、90° 单关节 axis 注入、checker、四宫格与像素门禁，没有降低阈值。唯一相机由正常运动先确定；错误失败后不重选该关节的视角、framing 或端点。使用真实 Chromium 143 + CPU SwiftShader，CUDA_VISIBLE_DEVICES 为空、单线程数值库和 nice=10，未占用训练 GPU。每个物体只保留一组 clean/axis。

新增拒绝原因按物体计数（同一物体可有多个原因）：

- `axis_limit_region_too_small`：1
- `axis_limit_trajectory_difference_too_small`：2
- `axis_mid_region_too_small`：2
- `axis_mid_trajectory_difference_too_small`：2
- `candidate_limit_clipped`：1
- `candidate_limit_motion_too_small`：3
- `candidate_mid_motion_too_small`：3
- `geometry_outside_frozen_camera`：1
- `normal_motion_or_framing_gate_failed`：13

结构拒绝的原始原因及历史筛选拒绝记录保留于容量审计和 private/*screening.json。

## 环境兼容诊断

源转换使用系统 numpy 1.26.4 / trimesh 5.1.0；原渲染 venv 中 numpy 2.4.2 / trimesh 4.7.1 会对等价重合顶点选择不同索引，首次严格拓扑测试出现 7 项不一致。没有修改原始数据、checker 或容差。用原转换依赖重读时，10 物体的全部顶点与 faces 均与保存 spec 完全一致；随后同一 Chromium 143 / SwiftShader 的原审计全部通过。初始失败、完整规范环境审计结果均保留在 private/source_environment_diagnostic.json 和 private/native_source_audit.json；不把兼容失败称为几何损坏，也不隐去失败记录。

本机复核源真实性时，使用系统 Python 的 numpy/trimesh 配合同一 pinned Playwright 模块，可在独立 /tmp 目录链接 `../TurnItOver/.venv-local/lib/python3.12/site-packages/playwright`，将该目录置于 PYTHONPATH，再运行新增浏览器测试；不修改共享 venv。图像重放仍使用原 `.venv-local` 和原浏览器缓存。初始 10 个样本来源先渲染后补做该真实性审计；全数通过后已把同一门禁加入生成器，后续生成在写入图板和审核行之前执行。

## 一致性与验证

原始 v2 的 413 个文件逐文件 SHA 一致，改动 0；继承的 392 个图像/程序文件字节一致；旧 56 条 Gold、checker 和人工记录逐条与原始 JSONL 行字节一致。训练清单和私有训练答案已扩充为76条，其中继承的旧56条逐行字节与v2一致。逐文件结果见 `private/continuation_audit.json`。

旧版 v2 manifest 固定的是历史生成代码 SHA。本轮最小修改了续建、正常相机选择及导出文档路径；旧代码快照存于 `private/provenance/` 并与原始 SHA 核对；另存生成时的续建脚本快照，真实性门禁完成后已补入生成器供后续续建直接拒绝失败来源。v2 manifest 和其全部数据保持不变；v3 单独固定当前代码 SHA，不把历史代码变化当成数据变更。由于 v2 固定历史脚本 SHA，用当前代码直接对 v2 运行旧 finalize 会按设计拒绝；旧 7 项单元测试是在其原脚本快照下复核，未绕过或改写 v2 的输入 SHA。

全量图像重放：76 样本、456 图像文件全部 SHA 相同，checker 与像素门禁重跑通过。10 个新来源均通过独立 URDF FK 对照真实浏览器导出顶点/拓扑的审计，最大顶点误差 1.33e-7 m，小于原阈值 1e-5 m。测试结果及执行命令见 `private/test_results.json`，原始重放摘要见 `private/replay_verification.json`，其 source_manifest_sha256 对应 private/replayed_manifest.json；报告封存后 manifest SHA 会更新，已重放文件的 SHA 未变。
