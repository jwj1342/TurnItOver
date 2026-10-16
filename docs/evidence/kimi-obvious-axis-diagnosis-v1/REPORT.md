# Kimi 明显轴错误固定证据诊断

本实验只使用上一轮 3 个经人类盲审确认、且与独立 Gold 一致的
`kinematics.joint_axis` 案例及其同资产 clean 对照。每例只调用一次真实
`k3-256k`，没有重试、主动取证、Prompt 调整、扩采或训练。

## 结论

- 3 个轴错误的严格诊断为 **3/3**。三例都同时正确给出 `fail`、
  `kinematics.joint_axis`、目标 joint、直接 moving part，以及人审确认的
  mid/limit 证据步 `[2, 3]`。
- 3 个 clean 的 `fail` 误报为 **0/3**：2 例 `pass`，1 例 `uncertain`。
  `uncertain` 的柜门案例明确说明另一个未测试关节不在当前证据覆盖内，因此按本实验
  预设不算误报，也不强行算作 Gold 的精确 `pass` 命中。
- 按预先定义的五个结构化字段，没有出现“人类能看懂、Kimi 却无法完成严格诊断”的
  案例。不过抽屉案例的自然语言解释只笼统说运动方向/轨迹不同，并把 prismatic
  `joint_1` 的修复误写成 “rotation axis / outward swing”。因此若把机械术语准确性也纳入
  “解释正确”，仍有 **1/3** 个部分正确的边界案例；它不改变五字段 3/3 的统计。

| 资产 | clean | axis 严格诊断 | 轴错误字段 |
|---|---|---|---|
| PM_102257 flip lid | pass | 正确 | fail / joint_axis / joint_0 / link_0 / [2,3] |
| PM_12036 cabinet door | uncertain（无误报） | 正确 | fail / joint_axis / joint_0 / link_0 / [2,3] |
| PM_25308 drawer | pass | 正确；解释有 prismatic/rotation 术语瑕疵 | fail / joint_axis / joint_1 / link_1 / [2,3] |

## 协议与审计

每个请求包含同一视角下 reference/candidate 的 rest、mid、limit 六张原生
256×256 图片，以及 Chromium 查询得到的合法 hierarchy。图片直接从已提交的人审图板
裁出，没有重新渲染。可见请求不含 Gold、腐蚀参数、seed 或人类答案；只包含公共 taxonomy、
结构化 verdict schema 和统一的部件定位规则。

总计 6 次真实请求，6 次都正常返回且通过 schema，共 17,532 input tokens、10,234
output tokens、27,766 total tokens。网关实际计费未知，`billing_cost_rmb` 保持 `null`。
独立重审验证了每例恰好一次调用、六张输入图 SHA、固定 `[0,1,2,3]` 历史、终判请求、
原始响应保留和源图板 SHA；6/6 通过。旧 Episode 审核器按文件名排序 reference，重审时在
临时镜像中把 `reference-rest/mid/limit` 规范为编号名；原始请求文件和图片字节未改动，
适配说明保存在每例 `audit.json`。

完整聚合结果见 `summary.json`，逐例原始 prompt、visible request、`images.json`、
`response.txt`、Episode、审核与私有字段评分见 `cases/<case_id>/`。复现和离线重审命令见
`REPRODUCE.md`。
