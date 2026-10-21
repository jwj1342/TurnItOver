# API Pipeline 原始记录独立审查

日期：2026-10-21。审查对象为 `7a73c91d98ef294e590171e59aa8c00dc2fee3c3` 工作树及原始 output。**总体结论 WARN；本轮有界审查已完成，same-family / provisional。** 未发现已核验记录中的伪造 GT、自归一化、结果数字不符、补丁被替换或失败调用漏计；这不是供应商真实性认证，也不支持 Active 的一般优势。

本报告由新审查代理 `/root/record_integrity_review` 直接读原始代码、请求、响应、PNG 和结果后形成。启动请求指定 `gpt-6-astra / ultra`，模型身份依据仅为 spawn 请求元数据。本轮未调用付费 API、未改原始记录或实现；旧 [audit](api-pipeline-audit.md) 的 ERROR/incomplete 保持为历史状态。本轮结论与旧代理中断记录分开。

## A–F 结论

| 项目 | 状态 | 原始证据及判断 |
| --- | --- | --- |
| A. GT 来源与隔离 | PASS，有范围限定 | `turnitover/api_pipeline/fixture.py:28–38` 由独立明确规格构建参考，未读取模型输出；`paired.py:25–37` 先渲染参考再调用 Generator。实际参考源码与参考编译器逐字相同，API 响应提取源码与 shared/final.ts 逐字相同且不同于参考。`fixture.py:52–72` 的 c00/c01/c02 可逐字重建实际 initial.ts。`scoring.py:119–154` 在 API 完成后读私有参考和标签；实际 Verifier payload 只有公开上下文及真实历史前缀，Generator 只另见候选源码和公开反馈。 |
| B. 计分与分母 | PASS | `scoring.py:26–34` 直接计数 invariant，没有用预测最大值归一化。`gt_validator.py:23–38`、`checkers/evidence.py:26–38` 测 rest、两个关节各自 mid/limit，共 5 状态；`geometry.py:29–43` 是 3 部件 × 5 状态 × 2 几何指标=30，另有 2 runtime invariants，总数32。contract 是额外布尔门，并不占32分母；`code_correct` 同时要求 contract、geometry、runtime 通过。 |
| C. 文件存在、数值与成本 | PASS（本地记录层） | 本轮新脚本 [check_records.py](../results/2026-10-21-api-record-audit/reviewer-trace/check_records.py) 独立检查 609 项，失败0；逐请求解码图像和响应、独立拼接15份 SSE、对照九台账。44次尝试、39完成/5失败、145467已报告 tokens、4次usage未知，与 `results/2026-10-21-api-pipeline/summary.json:2–12` 相符；没有货币费用记录。 |
| D. 路径确实执行 | PASS（记录及本地重放层） | `repair/iterative.py:178–240` 新目录调用 verify；`verifier/runner.py:113–122` 每次新建 ObservationSession。实际 c02 补丁响应只有 X→Y 一行，独立逐字应用结果等于 proposed.ts、下一轮 candidate.ts 和 final.ts；c00 Fixed 原始 stop 响应保持源码不变。另直接检查本轮执行者重放脚本及输出：`results/2026-10-21-api-record-audit/check_records.py:84–115` 的258项检查通过，22张观察PNG逐字节重放一致，六组最终质量及初次诊断/见证一致。未把脚本 fixture 测试当 API 效果证据。 |
| E. 范围、预算与因果归因 | WARN | `docs/api-pipeline-report.md:9–13,46` 已作主要限定；仍须明确 **1次 API 生成 + 2种受控源码注入**，不是3次独立生成或3个物体。只有一套三盒体、每组一次采样；协议经先前调试，c02 分段恢复。Fixed K=4 的计划未覆盖 door；没有等 token 或无诊断反馈对照，无法隔离主动选择或诊断对修复的独立贡献。 |
| F. 评价类型 | PASS，`simulation_only` | 独立作者规格/程序参考、真实 Chromium/Three.js 仿真和存档 API 交互。不是 dataset-provided real_gt；参考不是模型输出，故不是 model-generated synthetic_proxy；本报告图片阅读也不是盲评人类实验。证据：`fixture.py:10–38`、`paired.py:21–37`、`gt_validator.py:16–38`。 |

## 原始链路与真实结果

参考源码 SHA256 为 `c29399d9a75de19ebe9cff1c6f96b68e7a44dd26b888f1b2b6262b9e2f51d8e4`；API 基础源码为 `acde36209a8c6b03f41bc309551b267589be3dca9ca9152a84fbbc3748a5ae0c`。基础源码保留真实 API 输出中的 TypeScript 类型、命名 Mesh 和原生 joint set 实现；c01/c02 只在其上加受控 wrapper。c01 的 `initialX + 0.15` 与 c02 的 `(1,0,0)` 轴都在实际 initial.ts 中，未拿参考程序替换候选。

| 案例 | 组别 | 初次诊断 | 最终有限检查 | 观察动作 / 观察PNG | API尝试 / 已知tokens |
| --- | --- | --- | --- | --- | --- |
| c00 clean | Active | pass，正确 | 32/32，contract通过 | 4 / 3 | 5 / 16983 |
| c00 clean | Fixed | fail，误报 | 32/32，contract通过 | 4 / 3 | 2 / 11326 |
| c01 offset | Active | pass，漏检 | 22/32，contract通过 | 4 / 3 | 5 / 15697 |
| c01 offset | Fixed | pass，漏检 | 22/32，contract通过 | 4 / 3 | 1 / 4302 |
| c02 axis，分段 | Active | fail，正确定位 | 修复后32/32，contract通过 | 8 / 7（3+4） | 12 / 42939，另1次未知usage已含于12次 |
| c02 axis，分段 | Fixed | pass，未观测door | 28/32，contract通过 | 4 / 3 | 1 / 3449 |

初次全案例标签检测/类型集合匹配为 Active 2/3、Fixed 0/3；最终程序有限检查通过为 Active 2/3、Fixed 1/3。此处两个“2/3”含不同性质的成功：clean原本正确，c02经一次实际补丁修好。不能表述为“修复成功2/3”。全部有缺陷初始程序中，Active 仅修复 c02，c01未触发修复。`accepted=true` 是 Verifier 决策，c01 两组 accepted 与独立失败并存，不是完整正确性标签。

所有模型调用的 request.json 图像经 base64 解码后与调用目录拷贝、Verifier 本地 images.json 及真实轨迹对应的 PNG 字节一致；prompt 中 history 严格等于已经发生的轨迹前缀，remaining=4−前缀长度。无未来观察、私有参考源码或评分内容。Generator 的两次修复请求也绑定到了实际候选、公开反馈和被引用截图，而非人工替换后的输入。此结论是对存档输入的核验，不宣称本地记录具有不可伪造的外部签名。

## 图像语义与具体误报

本轮直接打开12张原始PNG，其中包括两个非空 findings 的全部引用、门修复后动作和偏移案例。图片阅读是本审查代理的语义判断，保持 provisional；与几何代理分开。

**c02 Active 的阳性解释得到可见支持。** `output/api-minimal-v6/c02/active/round-000/verifier/result.json:194–205` 引用 step 3。其 [front初态](../output/api-minimal-v6/c02/active/round-000/verifier/observations/000.png) 为直立红门，所引 [step 3](../output/api-minimal-v6/c02/active/round-000/verifier/observations/003.png) 为水平细条，宽度保留而高度消失；这确实支持绕水平轴而非要求的Y轴旋转。补丁 `round-000/visual-revisions/revision-000/response.txt:2` 将实际 override 的 X 轴改为Y轴。修复后 right视图中的 [step 1](../output/api-minimal-v6/c02/active/round-001/verifier/observations/001.png) 与 [step 3](../output/api-minimal-v6/c02/active/round-001/verifier/observations/003.png) 显示门从侧边细条打开为竖直板，和修复方向一致。它不能从图像单独精确证明所有米制尺寸/anchor。

**c00 Fixed 的引用存在，但解释错误。** `output/api-paired-v6/c00/fixed/round-000/verifier/result.json:201–209` 将屏幕向左移动解释成 world −X。实际 [step 2](../output/api-paired-v6/c00/fixed/round-000/verifier/observations/002.png) 与 [step 3](../output/api-paired-v6/c00/fixed/round-000/verifier/observations/003.png) 是相同斜视相机，正常 +Z 运动也投影到屏幕左侧。API基础源码 `shared/final.ts:64–72` 明确只改Z；本轮 world-geometry 重放显示 delta≈(0,0,0.3)，与参考相同，故不能把屏幕方向认成世界错误轴。Generator 的原始响应为 `{"stop":true}`，结果不是实际修复失败或程序被改坏。

**c01 是有可见偏移仍漏检。** Fixed 的 [front](../output/api-paired-v6/c01/fixed/round-000/verifier/observations/001.png) 显示蓝抽屉相对clean更靠右，并与门发生投影遮挡；Active 的 [top](../output/api-paired-v6/c01/active/round-000/verifier/observations/003.png) 也并非偏移不可见。初始源码 `output/api-paired-v6/c01/initial.ts:104–109` 和独立重放均为抽屉X从−0.25到−0.10，运动方向本身仍为+Z。Active摘要称 rest geometry consistent（`verifier/result.json:193`）过度概括；动作覆盖不等于认识了这个位置错误。

## 必须保留的协议限定

1. **K按动作计，不按全部视觉证据或计算计。** 公共精确 task、参考PNG、候选部件和关节ID、taxonomy与阈值均免费提供；stats动作无PNG，view/actuate各返回PNG。初始render-gate图不进入Verifier。证据：`verifier/runner.py:73–78,116–120`，`policies.py:72–86`，`policy/loop.py:28–48`。初轮两组均4动作/3观察PNG；c02修复轮为4动作/4观察PNG。Active额外决策API已计费，K平等不代表等tokens、等调用或等总观察。
2. **最终诊断协议相同，但过程不同。** `policies.py:66–83` 中 Fixed按预定动作执行后只调用一次最终模型；Active按每个前缀进行调用，再使用同一 final_prompt。调用是独立单消息请求，未把之前的assistant推理作为额外对话历史。Fixed顺序由 `policies.py:21–35` 固定为 stats/front/oblique_fr/drawer_limit，K=4不到door。c02差异不能归因于在相同证据下Active诊断更好。
3. **全案例检测指标和模型pass语义不同。** `final_prompt.txt:8` 允许针对已观察范围的pass；Fixed c02明确在limitations承认未观察door，但 `scoring.py:85–87` 按全案例缺陷标签判为漏检。这是全案例覆盖/检出指标，不是发现“已看到缺陷却错误解释”的证据。c00 Fixed和c01的错误则有实际观察基础。
4. **不能证明诊断反馈对修复必不可少。** Generator获得精确公开轴/anchor规格和完整候选源码；其源码同时含原本正确的Y轴实现及人为X轴override（`output/api-minimal-v6/c02/active/initial.ts:74–80,103–109`）。本次只有一次修复，无task+source-only对照。可说“依据包含正确诊断的公开包完成补丁”，不可说“诊断是修复成功的唯一原因”。
5. **32/32是有限测试通过率。** 几何阈值为5mm Chamfer和0.9 AABB IoU，每部件采样2000点、seed=0（`geometry.py:17–21`）。仅rest及逐关节mid/limit，runtime在reset后取值；没有覆盖任意中间角度、组合关节状态、材质、碰撞、刚体物理或所有运行时状态，也不是32个独立样本。样本范围报告已经明示，不能把它升级为模型整体准确率。
6. **恢复保持本地等价，随机供应商结果不保证等价。** v5失败request与v6续接request除 `stream`/`stream_options` 完全相同；prompt、图片、context/history、程序、公开采样配置和超时相同；没有新增Active观察/修复。原树文件hash全匹配；拷贝后仅6个汇总/报告文件改变，新增最终调用和Fixed记录另存。11次旧尝试加1次新Active最终诊断=12；旧524未重计或丢弃。证据：`recovery.py:36–93,108–130,158–160` 及新 trace 的 `recovery_changed_files`。temperature=1、跨日时间/缓存和SSE变化使其仍是明确分段恢复。
7. **“同模型”只达到本地可见配置和返回ID一致。** public_config不保存endpoint/credential指纹（`api.py:55–57`）；`client.py:162–186` 的request记录刻意不含URL/headers。所有成功原始响应返回 k3-256k，比较阶段max_tokens=8192、temperature=1；共享生成max_tokens=4096。这不能证明跨进程使用同一endpoint、网关路由、后端权重或种子，更不能审计实际收费。真实API请求在本报告中意指具有完整本地请求/响应原始链路，不是外部供应商认证。

## 可支持的陈述与后续处理

可支持：本次受控仿真中，API存档生成了一份不同于参考的候选；两种故障按记录注入；Active发现并正确定位可见门轴错误，真实补丁后新观察与有限检查通过；Fixed对clean误报，双方漏检offset；成本和恢复历史可逐项核对。Active 2/3与Fixed 1/3最终有限检查通过是这六组记录的描述统计。

不支持：三个独立生成任务的成功率、Active普遍优势、等总成本下效率优势、完整物理正确性、无人工规范帮助的照片重建、诊断对修复的独立因果贡献、货币费用已知或模型权重固定。无需改代码或重跑付费实验来修饰现有成绩；今后若要提出这些更强结论，应先冻结多对象/多种子协议、覆盖两关节的Fixed计划和相应归因对照。

本轮原始记录与视觉语义审查的缺口已得到有界补全；same-family/provisional的独立性限制继续保留。新机器报告见 [api-pipeline-record-review.json](api-pipeline-record-review.json)，复算及输入hash见 [本轮trace](../results/2026-10-21-api-record-audit/reviewer-trace/record-checks.json)。旧审查文档无需覆盖或改写为PASS。
