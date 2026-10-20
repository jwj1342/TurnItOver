# PartNet-Mobility Level 0 v4：60个独立物体、120条合格样本

日期：2026-10-20。**已达到60个独立物体、120条合格样本：clean60条、axis60条，待审核0条。** 新增仅进入Train；不训练、不推理、不使用付费API。

| 指标 | 数量 |
|---|---:|
| 本次获取的官方来源档案（累计） |400|
| 排除已有来源 / 真正新增档案 |160 / 240|
| 新增结构通过 / 拒绝 |142 / 98|
| 来源、许可、身份和几何去重后候选 |142|
| 本轮已渲染筛选物体 |58|
| 本轮自动通过物体 / 样本 |22 / 44|
| 本轮拒绝物体 |36|
| 合计自动通过物体 / 样本 |60 / 120|
| 最终已人工确认训练物体 / 样本 |60 / 120|
| 本轮人工通过物体 / 样本 |22 / 44|
| 新增待人工审核样本 |0|
| 尚未渲染的已验来源 |84|

来源保持原官方 sapien-sim/PartNetMobility 固定 revision `ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f`、原许可记录和非商业研究/教育用途。下载与转换使用现有脚本，排除旧160原始ID；原57筛选及另外已有失败来源不再计入新容量。全部142结构候选逐个验证档案/规格SHA、原生URDF规格、标注和rest几何。与旧94结构物体、已用Train38、Validation7、UniPhys正式5和冻结Test20按原始身份、可用model ID、内容SHA及原有精确几何指纹排重；未发现新队列重复。缺失跨数据集model ID和不同坐标/网格表示仍是精确指纹排重的明确限制。详见 SOURCE_AUDIT.md 与 private/capacity_audit.json。

严格复用现有生成、90°单关节axis注入、四宫格、原生256×256、checker及阈值；唯一正常视角、framing和端点在错误渲染之前确定。每个新物体只保留一对clean/axis。每个自动通过来源还运行原生URDF FK/拓扑与浏览器导出对照，原容差1e-5 m不变。系统numpy1.26.4/trimesh5.1.0与原转换一致，复用Chromium143和CPU SwiftShader，CUDA_VISIBLE_DEVICES为空、单线程数值库、nice10；不争抢训练GPU。

## 人工审核与资格

用户于2026-10-20针对本轮新增44条明确确认：“已全部人工审核，都没问题，复核预期”。确认范围包含运动差异清楚、mid/limit证据、rest一致及无出框/遮挡干扰。审核者记为user，日期精确到天，无虚构具体时间。代理仅转录实际总体批准；逐例运动描述由AI查看图板整理并标注来源，clean/axis结论按明确总体接受的成对设计映射，不声称用户另行提供了独立盲审标签。

确认原文、44个case_id和审核页面SHA见 private/user_review_confirmation_round2.json；审核前不含Gold的页面保存为 private/review_page_before_round2_confirmation.html。既有v3确认与审核记录不变。正式清单 train_manifest.jsonl 共120条，private/train_targets.jsonl 共120条；每个物体的clean/axis均经原门禁通过，review_manifest.jsonl不直接用于训练。复核导出命令为 `python -m scripts.finalize_level0_axis_v2 --out docs/evidence/partnet-mobility-level0-axis-v4`。

## 一致性、拒绝及验证

v3的580个文件逐文件SHA不变；继承的532个图像/程序文件和旧76条Gold、checker、人工记录、训练清单及私有答案原始行字节不变。原v2文件亦全部不变。历史代码输入SHA保留在基线manifest和private/baseline_source_pins，不改写历史manifest以迎合当前代码。

新增拒绝原因按物体计数，可多因：

- `axis_limit_region_too_small`：5
- `axis_limit_trajectory_difference_too_small`：8
- `axis_mid_region_too_small`：4
- `axis_mid_trajectory_difference_too_small`：11
- `candidate_limit_clipped`：7
- `candidate_limit_motion_too_small`：20
- `candidate_mid_clipped`：2
- `candidate_mid_motion_too_small`：21
- `geometry_outside_frozen_camera`：7
- `normal_motion_or_framing_gate_failed`：23

已通过16项单元测试、覆盖全部22个新来源的浏览器真实性测试、最终checker/资格/manifest验证；120条样本全量重放的720个图像文件SHA全部相同。重放时manifest归档为 private/replayed_manifest.json，其SHA与 private/replay_verification.json 中 source_manifest_sha256 一致；后续报告封存仅更新元数据清单，所有已重放图像/程序不变。具体日志与结果见 private/test_results.json。原v3历史测试/审计记录另外保存在 private/provenance/v3_*，本轮新日志为 private/test_logs/round2_*。最终120条均已通过真实性、checker和实际用户人工审核资格门禁；目标缺口为0。

后续续扩初始化的最小修复保留新capacity审计及其来源报告，避免被基线同名文件覆盖；新增回归测试通过。本批实际生成源码归档为 private/provenance/round2_generation_extend_level0_axis_v2.py，生成SHA与修复后runtime SHA分别保留，图像和程序字节不变。
