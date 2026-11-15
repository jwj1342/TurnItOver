# RP29 Table 1：OpenHands 修复与独立评分

## 当前入口与论文边界（2026-11-15）

`run_rp29_repair_batch.py` 默认使用 **OpenHands SDK 1.62.0** 的 `Agent` /
`Conversation` 执行修复。SDK 负责生成、工具解析、历史与执行；项目只提供一个
受限程序工具（读取当前程序／提交 exact edits）和每次提交后的确定性 gate、
新观测、judge 反馈及私有审计。没有把 SDK 的一次模型调用包回旧的 patch 循环。
`--harness legacy` 保留旧轨迹复查入口，两种 harness 的结果不得混为一组。

对应当前 Paper §4.3、§5.1 和 Table 1 的最小修改为：

1. 八个 judge 条件共用当前程序与固定观测接口；zero-shot sweep 与
   Ordinary/Evidence 复用识别任务已有的逐观测推理与确定性合并。
   三态各调用一次，每次仅接收该态的 reference/candidate 图像对及 runtime；
   共用提示词开头为 `Compare the supplied observations.`。
2. 每个新程序重新执行原生重放，保留 runtime；finding 引用的视觉证据以完整
   reference/candidate 图像对送入 coding agent，runtime 引用保留为文本。
3. 用同一个 OpenHands 程序工具和预算口径运行所有条件。私有 reference 源码、
   labels、审计结果及 evaluator 路径不提供给 agent；没有通用终端或文件读取工具。
4. 按冻结的逐缺陷 exposure 分类；不同视角造成的差异不直接算 motion。
   probe、temporal、unknown 单列，不能塞进 visible-at-rest。
5. 按 label index 接独立 checker，支持一例多个缺陷和修复引入的 regression；
   最终 Success、逐缺陷 Resolved、首次成功成本分别记录。未知结果保持 null。

这五项的接口和回归测试不等于完整 RP29 评分已经完成。默认的
`turnitover.repair.native_evaluator` 已接通 29 类开发评分路径，复用现有
原生浏览器、几何对齐、三角形／拓扑、材质、时序与资源检查；缺少原生范围、
对应关系或有效证据的具体案例仍保留 unknown。
`snapshot_scoring` / `candidate_scoring` 的诊断 F1 不能替代程序正确性。
正式评分须提供经过冻结的完整原生 evaluator。预检逐项列出
缺失项，并枚举 29 类的 checker 覆盖，未具备条件时不会标为 `experiment_ready`。

## 环境与模型配置

使用独立 Python 3.12 环境安装 `requirements-openhands.txt`；SDK 的 Pillow>=13
与旧渲染环境的 Pillow==12.1 冲突。通过 `--native-python` 指定现有原生重放环境，
无需重建或升级原生浏览器环境。`--native-root` 仍指向实际 RP29 ABI 的仓库。
运行配置分别归档 agent 与 native Python 环境的解释器版本和包版本；两种环境
均进入比较协议哈希，不能仅凭同一个解释器路径认定环境始终未变。

评分运行在 agent 环境中，`requirements-openhands.txt` 显式包含 Trimesh 与
`networkx`；后者用于面朝向修复的延迟加载路径。默认 evaluator 的预检实际执行
一个面朝向不一致的小网格检查；依赖不完整会在渲染和模型调用之前阻止批跑。

所有 OpenHands 条件使用同一个程序工具：首条消息和每次读取最多提供 64,000
个源码字符，超过时明确给出总长度、当前范围和下一段位置；`read` 支持
`offset` / `limit` 和可选的字面量 `query`。完整源码仍保存在任务中，exact edits
始终作用于完整当前程序；不删改大型几何数组，也不向模型开放私有文件。
工具读取同样计入原来的 agent-step 上限，不增加修复轮数。输入策略进入运行
配置与比较协议哈希；历史整段输入轨迹单独保留，不与新策略混合统计。

评分的性能修复只在本次检查内复用逐字节相同的网格、feature 和接触对测量，
并把逐部件表面相同比较移到接触对循环外。精确点面距离按三角形包围球半径的
二次幂分组查询，避免少数大三角形扩大所有小三角形的候选范围；仍保留逐面
保守下界筛选与全局最近面 tie 顺序。评分阈值、采样量、工作预算和 unknown
规则不变。大网格的拓扑／连通分量检查仍可能消耗数十秒，不能把这项工程
提速视为完整 29 类评分已冻结或正式 Table 1 资格已通过。

Generator 沿用 `TIO_GENERATOR_*` 配置。Kimi 配置示例不含密钥：

```dotenv
TIO_GENERATOR_PROVIDER=openai_compatible
TIO_GENERATOR_MODEL=kimi-k3
TIO_GENERATOR_BASE_URL=https://api.moonshot.cn/v1
TIO_GENERATOR_API_KEY_ENV=MOONSHOT_API_KEY
TIO_GENERATOR_STREAM=true
```

`MOONSHOT_API_KEY` 从私有环境提供。Claude 可使用 `anthropic` 或已确认的
compatible endpoint，模型 ID 必须显式配置。SDK 必须识别 generator 的图像能力，
不能静默丢掉引用图像。`kimi-k3` 尚未被 SDK 能力表识别；核实实际 endpoint
支持图像后，显式设置 `TIO_GENERATOR_SUPPORTS_VISION=true`，该声明会归档，
声明本身不代替实际 endpoint 的视觉验证。OpenHands 当前不接旧客户端的跨进程发送间隔和 thinking
override，配置这些选项会明确报错；不能假定旧客户端的限速配置对 SDK 生效。

每个模型 judge 用独立 `--judge-env CONDITION=PATH`，文件使用 `TIO_JUDGE_*`
字段，记录模型／checkpoint 与 endpoint 哈希。条件名与论文行对应如下：

| 条件名 | 论文 judge | 视觉输入 |
| --- | --- | --- |
| none | 无 judge，固定修复轮数 | task reference |
| pixel | 训练集冻结阈值的 pixel difference | 三态 sweep |
| zero-shot-single | Qwen3.5-4B zero-shot | zero 图像对 |
| zero-shot-sweep | Qwen3.5-4B zero-shot | 三态逐对推理并合并 |
| zero-shot-claude-sweep | Claude Fable 5.1 zero-shot | 三态逐对推理并合并 |
| ordinary | TurnItOver 2B Ordinary | 逐对推理并合并 |
| evidence | TurnItOver 2B Evidence | 逐对推理并合并 |
| gold | 私有当前缺陷 upper bound | 三态 sweep |

single 条件只接收 zero 图像对及 zero 状态的 runtime，不从文本中获取 mid/limit
状态的运动证据；其余条件保留各自的固定 sweep 输入。
当前 OpenHands 实验路径始终提供 `public_record`，模型 sweep 共用相同的三次
输入和合并函数。旧的无 `public_record` 简化调用分支仅供历史兼容，未在本次改动。
输入一致并不自动成为训练消融：训练效应仍须比较同基座／规模模型的未训练与
训练 checkpoint，且保持这些输入、合并和解码配置一致。

名字本身不证明部署的是正确 checkpoint；必须核对服务端版本并归档配置。
无效的结构化判定最多额外重试三次，仍无效则 uncertain；全部已报告 usage
计入成本。传输失败不会被伪造为 PASS。结构化输入复用原 verifier schema。

新冻结输入会包含 `public_context`（公开 ABI IDs、任务、预算）。旧清单没有该
字段时，模型 judge 预检会要求重新冻结；不会把历史 runtime 或 assistant 目标
补成当前观测。现有输入文件及旧实验输出不在原位置覆盖。

## 无 API 调用的预检与显式预算

```bash
PYTHONPATH=. python scripts/run_rp29_repair_batch.py \
  --manifest /path/to/new-frozen/manifest.json \
  --native-root /path/to/native-rp29 \
  --native-python /path/to/native-venv/bin/python \
  --output /path/to/new-output --preflight
```

执行时必须显式给 `--rounds`、`--max-agent-steps`、`--repetition`。round 是一次
程序提交（含无效编辑／原样提交）；agent step 是 SDK 一步，两者不能等同。
none 的提前 finish 消耗一次原样提交，并继续同一个 Conversation 到固定轮数；
全程仍受同一个 agent-step 上限保护。其他条件可由公共 PASS、generator finish
或预算停止，私有 audit 永不控制停止。最终一轮同样重放、判定和独立审计。
SDK 对话与事件保留在输出目录，关闭 Conversation 不删除记录。

论文中的 repair 重复数还是 XX，具体修复预算也未冻结。本文不给它们设定论文
默认值；示例预检中的配置不构成正式实验协议。`--collect-unscored` 仅明确允许
收集未完整评分的轨迹，不能据此填写 Table 1。

## 接入原生独立 checker

默认使用 `turnitover.repair.native_evaluator`，也可用
`--evaluator-module package.module` 指向已审核的原生评分模块，导出：

- `SCORING_CONTRACT`：`status="frozen"`、`supported_defects`（29 个 taxonomy IDs）、
  `required_checks`（包括未注入属性的 regression 检查）、`exposure_status="frozen"`，
  以及实际版本／阈值和已审核逐标签 exposure 的出处。
- `make_audit(case, replay)`：返回 `audit(current_source)`。可以复用
  `table1_scoring.make_repair_audit`，按 label index 接现有检查器，验证当前源码哈希。
  case 中 `private_labels`、`private_exposure`、`private_exposure_history` 来自绑定哈希的
  私有清单，只用于评分；即便首次审计前失败，也保留逐缺陷的汇总分母。
- 可选 `make_gold(case, replay)`：返回 `gold(current_source, public_evidence)`，
  只在 gold 条件使用，输出通过原 schema 校验的当前 findings；已修复的缺陷不能
  继续照搬初始标签。gold 也不能获得私有 reference 源码来直接替换 candidate。

默认 evaluator 现已提供该 Gold 接口：每轮重新检查当前程序，只输出尚未修复的
原注入 taxonomy、公开部件／关节 ID 和通用修复建议，不输出源码、私有参数或测量值。
无法归因到原标签的新 regression 或缺少证据时返回 uncertain，不能伪造 PASS。
引用的当前步骤仅提供重放上下文，不据此自动补写或冻结 exposure。

缺失 predicate 不提供“默认通过”的桩函数。`score_audit` 保留逐缺陷和 program
checks 的 unknown；原始标签全部消失但引入新错误时，regression check 仍使
Success 为 false。完整覆盖标志与 pass 值分别记录，部分覆盖不能宣传为完整评分。
模块源码哈希和合同进入冻结运行配置，变更后拒绝续跑同一输出。

### 已接通的原生评分（开发验证）

重放桥接沿用现有 `ObservationSession` 和 `exportGeometry`，每次执行保存
reference/candidate 的当前 ABI 契约、三态实际命令和世界坐标三角网格。
导出位于独立的 `*-evidence.private.json`，绑定文件哈希、程序哈希及角色；
只传给审计，不放进 public runtime、judge 或 OpenHands 对话。

默认 evaluator 的候选路径覆盖：

- 全局／各向异性尺度和缺失部件；
- 部件位置、朝向、对称性和 pivot offset；
- 关节轴、类型、行程、锚点和方向；
- 三角面与 draw-call 预算（只能使用任务显式公开的 caps）；
- 开放体积、原生严格自交、重复面／部件与层级；
- 五类材质的 CPU 属性、纹理／UV 与实际 GPU program 读回；
- 真实资源销毁事件、计数与脱离场景的顺序；
- 原配方时间戳下的运动、相位和绝对时间历史一致性；
- source 绑定的薄片、倒角、悬浮连接与凸／非凸互穿检查。

几何和运动类复用现有 `GeometryAlignmentChecker` 的逐部件、逐状态不变量；
关节类型／行程按实际 ABI 独立计分，不因同部件的其他几何缺陷连带判未修复。
全程序契约、所有部件的几何对齐和预算
独立于注入标签检查，防止“修好了标签，但删了其他部件”被算成 Success。
缺失／空 reference、缺失状态、无效网格和过期导出不能成为默认通过。
同一部件上的多个缺陷可能耦合对齐判据，因此这些是待原始构造负例检验的
候选判据，不能声称已复现全部原生缺陷定义。

拓扑、层级、材质、生命周期、时序、特征形状及部件接触均有独立 regression
检查，覆盖三态以及私有的返回初态、原始 authored history；时序程序还检查同一
最终时间戳的短历史及该次销毁。额外私有检查不增加任何条件的模型观测。
缺失 GPU program、不可读取纹理／属性、未建立 reference 运动等情况保持 unknown。

薄片／倒角／悬浮／互穿所需的范围由 `native_scopes.py` 从有哈希绑定的原始构造
报告恢复，只读取 Train/Validation。wrapper 跟随原 collector 时还校验其预注册
对报告及两份原始程序的绑定；只取标识、组件范围和原阈值，不消费历史判定。
新冻结清单将其存入私有 `labels.private.jsonl`，旧清单不会被自动覆盖。
这四类在当前 Validation 的 17 例均已恢复范围（倒角 9、悬浮 3、互穿 3、薄片 2）。

评分合同 v4 保留精确恢复的快速路径，另外允许确定性的启发式检查，接受等形
细分和顶点重新编号，不强制 coding agent 逐顶点复原原程序：

- 薄片：在 reference 冻结的 PCA 轴上比较表面采样的分位厚度，比例范围
  0.75–1.25，并要求原位、原尺度双向表面覆盖；未使用的外框顶点不起作用。
- 倒角：使用 source 组件范围、局部表面距离和法线一致性，小面积法线组额外采样；
  可读倒角深度时容差取深度的 25%，否则用组件尺度的 0.1%，并保留原数值容差下界。
- 悬浮：实际三角形距离加相对接触面积覆盖，至少保留 reference 覆盖的 50%，
  避免用针尖触碰替代原装配面；不是重力支持或力学稳定证明。
- 互穿：认证凸体仍用完整分离轴检测；其他可测网格用固定三方向射线、7³ 占据
  网格和双侧表面采样。参考装配原有交叠可保留，但新增或明显加重交叠会失败。
- 重复部件：除重复面外，再计显著连通组件，覆盖微移后不再逐面重合的复制。
- 全程序 regression：几何用双向表面覆盖；材质属性保留原始哈希，并在哈希不同
  时比较面积积分的一／二阶及位置交叉矩，避免合法细分仅因 UV／法线编号变化失败。

这些方法复用现有 NumPy／SciPy／Trimesh 与原生浏览器，不增加执行 harness。
固定采样数为 512，随机种子为 0，表面默认相对容差为尺度的 0.5%、覆盖率为 95%；
完整默认值与工作量上限在 `SCORING_CONTRACT` 中版本化。它们是工程默认值，
尚未按 Train 校准，也未按 Validation 或模型修复成绩调参。射线歧义、缺少组件
对应、无法读取的属性或工作量超限仍保持 unknown。属性矩不是任意 UV 场的等价证明，
同部件的若干几何标签仍可能耦合；不能声称形式化物理正确性。
源码来源、工作树 SHA256 与新增方法见 `docs/native-scoring-provenance.json`。
因此默认合同仍为 `status=development`、`table1_eligible=false`；29 条路径不代表
全部案例与全部物理语义已审核，不得用单纯 Chamfer、bbox 或历史报告冒充通过。

评分合同 v5 修正公开预算接线：`2147483647` 仅为 context 占位值，不作真实任务
上限；`task_runtime_requirements` 中标记 `public_task_budget` 的明确上限参与评分。
draw-call 与三角形预算分别判定，仅给出其中一项时，另一项仍为 unknown；
不从修复后的 Validation 程序推断或拟合预算。冲突的明确上限会拒绝运行。

开发 pilot 可显式使用 `--collect-unscored` 并指定修复预算，采集轨迹和已有的
逐缺陷评分；该选项不跳过模型、预算、私有标签清单和原生环境检查。预检会列出
29 类路径、开发合同与逐项缺口，且不会初始化模型或发送 API 请求。

编译／渲染拒绝为确定性 gate 失败，记录 Success=false；此时不调用需要网格的
predicate，无法确认的逐缺陷状态保持 null。TypeScript `CompileError` 与浏览器
报告的确定性候选错误均走 gate，脱敏的编译诊断可返回编码模型继续修复。
浏览器断连、reference 执行失败和其他基础设施错误仍保留为中断，不能伪装为
候选缺陷或 PASS。

新冻结的 role-specific joint_range 案例记录
`replay_command_policy=current_joint_limits`：每轮依据当前程序的实际 limits
解析 zero／mid／limit，而非继续使用最初错误程序的角度。原配方和 provenance
保留；已有清单默认维持 `authored_numeric`，不会静默重写旧实验。
须重新冻结输入再运行该修正后的语义协议。

Exposure 分类只接受显式 `frozen=true` 的 label-step 映射；自动投影、尚未确认的
标注不自动升级为正式 exposure。分类必须使用原始 history 中的实际 step/state/view，
不能把原 history 的 runtime step 当作三态 pair 下标。

`scripts/summarize_table1.py` 离线读取两种 generator 的 `summary.json`，检查八行、
两种 generator、case/repetition 一致，分别列 Success，再平均两个 generator 的
Resolved 和成本。失败／中断保留在分母，缺失值保留 null；成本仅用最终成功
轨迹的首次成功迭代和 token（包括该轮 verifier 调用），不使用整个批次总 token 冒充 cost-to-success。
`table1_eligible` 与数值一并输出，部分评分不能当作正式论文结果。

## 本地验证与待冻结项

2026-11-15：最终单元回归得到 494 passed、6 skipped、2 deselected，覆盖 SDK、
批处理、29 类评分接线、候选编译／运行失败、证据隔离，以及 sweep 条件的提示词、
逐对图像和汇总一致性。SDK 集成测试运行实际的 `Agent` / `Conversation`，只替换
模型响应，不发起 API 请求；有一条 SDK 依赖的 Pydantic ReadOnly 提示，不影响通过。
其中本轮新增 26 项，覆盖等形重网格、局部倒角、厚度反例、非凸占据、针尖接触、
微移复制、原装配交叠加重、关节独立计分及当前程序 Gold；另复核完整审计 JSON 落盘。

原生浏览器检查另有 16 passed：12 项解析式程序验证实际 GPU 材质读回、空销毁
回调、薄片／悬浮／互穿、动画相位／累计时间错误及返回初态才出现的重复部件，
并验证合法细分通过全部 program checks、细分后的 UV 改坏仍失败及结果可保存；
4 项使用有原始 SHA256 绑定的真实 Validation 程序，分别验证薄片、缺倒角、
悬浮、互穿的原始损坏为 false、恢复 reference 为 true。Test 未读取，模型 API
调用为 0。这些是工程评分复核，不是全量 109／900 案例或 Table 1 的模型修复实验；
也不证明实际 Kimi／Claude endpoint 与全部原生缺陷语义已经验证。

两个排除项分别为使用 Linux `fork` 的跨进程测试，以及旧归档的原始字节哈希
测试。Windows checkout 把归档的 LF 转为 CRLF；只在内存中还原 LF 后，124 项
归档哈希全部匹配，未改动归档内容或将该检查改为默认忽略。

正式实验仍需冻结：29 类实际原生 predicate 与阈值、regression roster、审核后的
逐标签 exposure、修复轮数／agent 总步数、重复数与 seed 支持，以及实际
generator 和各 judge checkpoint。以上缺口不通过默认阈值、自动 exposure 标签
或将诊断 F1 当作程序评分补齐。

## 旧分支轨迹记录（legacy）

下文是 2026-11-14 的原始 none/pixel 轨迹采集记录。命令执行旧实现时须额外
指定 `--harness legacy --repetition <id>`，轮数仅代表当时工程设置，不是论文冻结值。

`prepare_rp29_full_repair.py` 冻结 v2.1 Validation 的全部 109 defective cases，
排除 11 clean；不读取 Test。每例清单绑定程序、reference 图与真实重放配方。
24 个缺失 supplementary 配方已从原 observation-plan／时间合同恢复；出处与哈希
保存在 `recipe_provenance`。3 个 joint_range 案例须按 `commands_by_role` 重放，
3 个 temporal 案例保留 `time_sequence`，不把同一数值命令强加给两个程序。

Train 600 全部是 defective。pixel 校准使用这 600 例和 600 个明确标记的
reference-self 合成 clean 对照，未使用 Validation 调参。当前阈值为 0；
它不代表真实 clean 泛化校准，像素相同也不代表 runtime／物理正确。

先进行不调用模型的输入预检：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_rp29_repair_batch.py \
  --manifest ../.run-artifacts/rp29-table1-full-20261114/manifest.json \
  --native-root ../TurnItOver-taxonomy-900 \
  --output ../.run-artifacts/rp29-table1-kimi-full \
  --preflight
```

明确选择未评分轨迹采集后执行：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_rp29_repair_batch.py \
  --manifest ../.run-artifacts/rp29-table1-full-20261114/manifest.json \
  --native-root ../TurnItOver-taxonomy-900 \
  --output ../.run-artifacts/rp29-table1-kimi-full \
  --env-file /path/outside/repository/kimi.env \
  --conditions none pixel --rounds 3 --collect-unscored
```

两组各运行一次，共 218 条轨迹。none 固定消耗三轮，pixel 可由自己的 judge
或 generator stop 提前结束。私有标签、reference 源码、归档 candidate runtime
不进入模型；none 只获得 reference 图、当前源码、公开任务和公开预算。
每个新源码哈希重新执行原生浏览器，pixel 使用新采集的三态配对图。

逐例错误会保留并继续后续案例；重启同一输出目录会跳过已有结果和中断案例，
不自动重试可能已计费的请求。输入、校准、配置或执行代码变化时拒绝续跑。
`config.json` 记录单次重复、模型、轮数和 seed 未请求／未核实状态。
`progress.json`／`summary.json` 记录完成数、错误、调用与提供方报告的 token 消耗。

**本入口未实现完整 29 类独立评分。** `status=complete` 仅表示轨迹采集完成；
Success、Resolved、Needs motion／Visible at rest 和 Cost to success 均不能据此填表，
相关评分保持 null。pixel judge pass 只控制自身停止，不充当独立成功标签。

验证记录：2026-11-14，109/109 个原始 defective 程序均完成真实原生浏览器
三态重放，0 API 调用、未用 GPU。记录在
`../.run-artifacts/rp29-repair-code-check-20261114/all109/summary.json`。
验证环境为 Playwright 1.58.0／已缓存 Chromium；可使用
`uv run --with playwright==1.58.0 python ...` 复用该版本，避免浏览器版本不匹配。
该记录只证明初始输入可执行，不代表模型产生的任意 patch 均可执行。

## 可选共享请求间隔与重试

模型客户端支持显式配置（默认均为 0，保留旧调用行为）：

```dotenv
TIO_GENERATOR_REQUEST_INTERVAL_SECONDS=60
TIO_GENERATOR_MAX_RETRIES=3
```

同一 base URL／密钥的所有已启用客户端跨进程共用文件锁，一次只允许一个请求
在途；重试也经过相同发送间隔。按请求开始时间计时，失败也占用时间槽。
429、502/503/504、连接中断和超时最多重试三次（含首次最多四次）。重试等待
取请求间隔、30秒基数的指数退避和服务端 Retry-After 的最大值；429冷却由所有
共享客户端遵守。403等非临时HTTP错误、非法JSON与空文本不自动重试。

状态与脱敏尝试日志位于系统临时目录的 `tio-request-control-<uid>/`；文件名
只含 endpoint/key 的联合哈希，事件不包含密钥、URL、prompt或响应正文。
所有相关角色都应设置相同间隔；默认未启用的旧客户端不会加入共享通道。
此处60秒仅为单例验证设置，不能据此声称已满足账号TPM；应依据实际额度设置。
超时请求可能已在服务端执行／计费，重试日志须与成功响应usage一起保留。

## 长推理的流式传输

`TIO_GENERATOR_STREAM=true` 为 OpenAI-compatible 接口启用 SSE，默认关闭。
客户端持续消费 reasoning／心跳，只把最终 content 送入原有修复解析；保留
提供方 usage，必须同时收到正常 finish_reason 和 `[DONE]`，截断不算成功。
该选项仍经过共享发送间隔和重试通道，不改变 prompt、模型或修复轮数。

2026-11-14 使用同一个失败第三轮请求排查：经环境代理出现TLS EOF；直接
非流式曾在126.2秒返回Cloudflare524；同样payload经代理非流式复核又在72.2秒
成功。经代理流式在5.3秒收到HTTP200响应头、28.7秒收到首事件，持续155.6秒
后完整结束，返回2209事件和usage。说明存在间歇性传输问题以及非流式长等待
的网关超时风险，不能把RemoteDisconnected一律归为RPM或模型输出错误。
直连也出现TLS握手超时，因此不作为默认方案。原301缺少Location证据，来源
尚未定位；不跟随未知跳转传递凭据。
