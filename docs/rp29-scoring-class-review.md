# RP29 评分按缺陷类别复核（2026-11-15）

## 范围与结论

按用户要求，沿用此前 **11 类**记录，只为其余 **18 类各取一个代表 case**。
没有运行 109 题全量评分，没有重跑已有 11 类，也没有调用 Kimi 或其他模型 API。
18 个新代表均完成现有原生回放与评分，无缺包、异常或诊断超时。
仅对本轮发现问题的 8 个代表复用已保存回放进行定向复核；重复零件另复核一次。
复核后，新增代表中 17 类的原始缺陷被检出（`resolved=false`），1 类保持 unknown。
这是开发评分的类别检查，不是正式 Table 1 结果，也不证明整批都能确定判定。

服务器使用用户指定的 `10.201.37.233`，实验目录为
`/91004t2/laq/code/ThreejsJudge/TurnItOver-table1-openhands-protocol`。
本轮先将该目录由 `7c048fc` 更新到 `1920060`；原有四处未提交修改已逐文件
三方核验为该提交所包含，保存 patch、文件快照与 stash 后才更新。
备份位于 `../.run-artifacts/table1-server-update-20261115-121148/`。
提交时接续目标分支新增的 `dec5653`，保留其主动三态选择改动。

## 最小修复

1. 从哈希绑定的原始构造报告恢复 `condition.target`、`condition.part`、
   `duplicate_target`、关节 ID 和 `target_subtree`。只补当前缺陷缺少的标识，
   不使用旧 verdict，不把一个缺陷的目标分配给其他缺陷；兼容原标签的 `params.joint`。
2. 几何对齐使用回放中**已经采集**的对应历史快照，缺件也检查这些快照，
   避免只看三个主状态而漏掉回零／历史触发缺陷。两侧历史不匹配继续 unknown。
   全局缩放／空目标的各向异性缩放按实际导出的 reference 网格检查，
   不再要求空的 ABI 根／辅助容器产生几何不变量。
3. 重复零件除了目标自身的拓扑，还检查该目标世界坐标三角面的重复次数，
   防止复制网格挂到其他 ABI 父部件时被误判为已修复。
   这项补充针对坐标精确重合的三角面，不是任意重网格几何的重复实体认证。
4. 在已有准备脚本中增加评分标识刷新入口，输出新的同目录 manifest 和私有标签文件。
   校验原数据、注释、程序、标签实例顺序及原输入哈希；保留图像、观测配方、
   exposure、公开任务和旧结果，不覆盖旧冻结文件。

评分合同版本升为 6，状态仍为 `development`，`table1_eligible=false`。
未改变阈值、表面采样数、拓扑工作预算、缺失证据的 unknown 规则或模型输入。

## 新增 18 类记录

评分时间包含读取私有证据和生成 audit；不含原生回放、进程启动或模型调用。
采用 3 个独立 CPU worker 并发诊断，因此是本轮观测到的单题墙钟时间，
不是串行基准或最坏情况。选择每类代表时优先已有回放，再选较小源码；
不能将这些较小代表的耗时外推为 109 题均值。
“检出”指未修复程序的该项 `resolved=false`，不代表整题其他检查已完备。

| 缺陷类 | 代表 case | 纯评分秒 | 回放秒 | 结果／调整 |
| --- | --- | ---: | ---: | --- |
| geometry.anisotropic_scale | rp29-anisotropic-history-ead331375e633c286a4f | 1.2260 | 1.3072 | 修复空容器与历史检查后检出 |
| geometry.duplicate_part | rp29-duplicate-history-fb8c5cc4be2d76edd062 | 1.1709 | 1.4740 | 恢复目标并补跨父节点重复检查后检出 |
| geometry.missing_part | rp29-d6c7519f71d85f70ba1f | 1.7737 | 1.4904 | 恢复目标并检查历史后检出 |
| geometry.open_volume | rp29-b57d64cbf145a8bedc8b | 1.8290 | 1.7705 | 检出，沿用首次记录 |
| geometry.self_intersection | rp29-component-crossing-39bbf81f3252310e08cf | 4.7511 | 1.1302 | 检出，沿用首次记录 |
| kinematics.direction_flip | rp29-e9c863242c79214ee306 | 1.3663 | 1.0006 | 检出，沿用首次记录 |
| kinematics.joint_anchor | rp29-anchor-gap-5ce538dd42719541fa04 | 2.9005 | 1.4614 | 检出，沿用首次记录 |
| kinematics.joint_axis | rp29-1eed899e29cc0c8e76f4 | 3.6593 | 1.7308 | 恢复目标／兼容关节参数后检出 |
| kinematics.joint_range | rp29-93aef13f3feff7114152 | 1.1718 | 1.1240 | 恢复关节标识后检出 |
| kinematics.joint_type | rp29-0622c498d276ffa5cb8b | 5.6121 | 1.8650 | 恢复关节标识后检出 |
| material.basecolor_as_rough | rp29-material-query-aaab8f4ca2bbf947d38f | 3.2611 | 2.0961 | 检出，沿用首次记录 |
| material.metalness | rp29-98e3bac851e97d852223 | 0.1593 | 1.3950 | 恢复目标后检出 |
| material.normal_map | rp29-normal-gap-0e299f37361d0f714926 | 0.2022 | 1.3754 | 检出，沿用首次记录 |
| material.tiling_scale | rp29-187fb5ab0597f54b3ec5 | 2.3083 | 2.0689 | 恢复目标后检出 |
| structure.part_rotation | rp29-split-transform-523e4ca36ed94f116fbe | 2.0742 | 1.1555 | 检出，沿用首次记录 |
| structure.pivot_offset | rp29-split-transform-e0ca164f33e7caf1d0d0 | 1.5357 | 1.1747 | 检出，沿用首次记录 |
| structure.symmetry_break | rp29-balanced-d6c6dd08354cd00a8322 | 6.3692 | 3.4424 | unknown：缺少可用目标绑定 |
| structure.wrong_parent | rp29-44287dd7f3e54e2c5172 | 6.0763 | 3.1394 | 检出，沿用首次记录 |

回放时间是每个代表首次采集的时间；修复后的定向复核均复用原回放，新增回放耗时为 0。
原始记录在 `../.run-artifacts/table1-class-review-20261115-v1/`：
`selection.json`、`baseline/`、`fixed/`、`duplicate-fixed/`、
`replays/`、`class-review-summary.json`。逐次 audit 和代码差异均记录 SHA-256。

## 沿用的 11 类与仍存在的边界

沿用 `1920060` 之前本会话实际检查的记录，不重新计时：
`geometry.global_scale`、`geometry.missing_bevel`、`geometry.thin_sheet`、
`material.emissive`、`runtime.draw_calls`、`runtime.missing_dispose`、
`runtime.tick_missing_or_phase`、`runtime.triangle_budget`、`structure.floating`、
`structure.interpenetration`、`structure.part_offset`。
此前存在过的 global_scale／interpenetration unknown 不因本轮覆盖到 29 类而变成通过；
全局缩放容器问题已用合成回归验证修复，遵照用户要求没有重跑旧代表。
大网格连通分量、自交、接触等仍可能消耗数十秒；保留上轮大样例耗时记录，
不能用本轮 18 个代表都小于 7 秒宣称长尾已消失。

对称性代表的原标签 `parts=[]`；哈希绑定报告未给出当前 evaluator 可直接使用的
逐零件选择器。其其他文档含命名组件，但本轮不根据 candidate 差异猜目标，
也不把整场景几何失败冒充该缺陷的独立失败，因此保留 unknown。
这不阻止完成类别性能复核，但仍是正式冻结评分前的一个具体缺口。

## 更新现有冻结输入的评分标识

本轮新清单位于原冻结目录的 `class-reviewed.manifest.json`，SHA-256 为
`1cbf1a1d7d014693eb875e1677a07fde72f43cc326a4d7eed0ba0c82721ce60e`。
原 `manifest.json` 的 SHA-256 仍为
`349aa89ad29dc4d9d2bc973648bfc801a5b9719789d5c0c90a8186cef1efca61`。
刷新 109 条**元数据**不执行其回放、评分或模型调用。

```bash
PYTHONPATH=. python scripts/prepare_rp29_full_repair.py \
  --refresh-scoring-scopes-from /path/to/inputs/manifest.json \
  --dataset /path/to/hash-bound-dataset \
  --output /path/to/inputs/class-reviewed.manifest.json
```

修订清单不能与旧 manifest／旧代码实验目录直接续跑或混合统计。
本轮未重启正式 none/pixel 实验，亦未改变此前要求的 10 步上限。

## 代码验证

提交前本地评分、作用域恢复、元数据刷新、Table 1 协议和真实 OpenHands SDK
离线集成测试共 **138 passed**；SDK 模型响应使用 mock，没有联网调用模型。
Linux 现有原生环境的单元测试阶段结果为 **525 passed、1 skipped**，跳过的是
该环境未安装的 OpenHands SDK 专属模块，已由前述独立 SDK 环境覆盖。
Windows 全套初查的两处失败分别是已有测试使用 Linux `fork`、归档快照受 CRLF
换行影响；同两项在 Linux 通过，本轮没有为此修改评分以外的代码或历史快照。
最终部署后再以服务器最新提交执行单元测试，结果保存在独立部署记录目录。
