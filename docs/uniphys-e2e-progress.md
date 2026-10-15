# UniPhys E2E：分支进度与后续工作

> 分支：`laq/sft-dataset`（合并 `laq/uniphys-e2e`）｜核对日期：2026-10-13
> 依据：[Issue #4 Roadmap](https://github.com/jwj1342/TurnItOver/issues/4#issuecomment-6098267539)及[训练与评测补充](https://github.com/jwj1342/TurnItOver/issues/4#issuecomment-6098859780)。本文记录已证实的实现，不把计划、脚本存在或 Verifier 接受等同于研究成果。

## 给后续 Coding Agent 的约束（必须遵守）

**每次修改代码后、结束本次任务前，必须同步检查并最小化更新本文档。** 只改动受到影响的进展、缺口、优先级和验证证据；不重写无关段落，不累积流水账。即使状态未变化，也要更新文末的「最近同步」一行，注明本次改动及「状态不变」。只有实现和对应验证均可核实时，才将事项写为「已完成」；未运行的测试、未执行的脚本及模型自报 `pass` 不算验收。保持下述任务清单最多五项，完成后删去或替换，不持续追加。

## 目标与现状

本分支要把现有 3D 资产变成**交互式 Verifier 的训练与评测数据**，最终验证：证据约束监督能否让小型视觉模型在有限观察预算下更好地选择视角/关节动作、定位缺陷并及时停止。主线是「资产 → 合格候选与独立真值 → 观察 Episode → 训练监督 → SFT → 留出评测」。**正式资产池现有40个独立物体；PartNet 35资产已完成第一轮普通/证据约束 SFT，但7资产 Validation 仍不足以形成泛化结论。**

**已实现：资产与运行入口。** [`download_uniphys.py`](../scripts/download_uniphys.py) 下载并校验 UniPhys-Bench Part 2 的指定资产，记录版本和文件哈希；[`render_uniphys_references.py`](../scripts/render_uniphys_references.py) 从 URDF 构造内存 `AssetSpec`，导出多视角参考图，并注明 continuous joint 固定零位、不计作已验证运动覆盖。[`run_uniphys_iterate_batch.py`](../scripts/run_uniphys_iterate_batch.py) 接入 Generator → Render Gate → Verifier → Repair，保留任务状态、调用与 token 统计。已有一次 20 物体真实模型批量运行，**4 个被 Verifier 接受**；这不是独立正确率，详细条件见 [`uniphys-e2e.md`](uniphys-e2e.md)。

**已实现：正式 Test 缺陷基准、批跑入口与评分。** [`prepare_uniphys_verifier_benchmark.py`](../scripts/prepare_uniphys_verifier_benchmark.py) 生成 **20 物体 × clean / 关节轴错误 / 部件偏移 = 60 案例**、私有 gold 和逐案例预检；正式产物位于 `data/benchmarks/uniphys-verifier-test/`，60 个候选程序 SHA 唯一。[`run_uniphys_verifier_batch.py`](../scripts/run_uniphys_verifier_batch.py) 在调用现有 Verifier 前校验 catalog SHA、test-only 划分、每物体三种变体及候选 SHA，逐案例原子记录进度，完成后调用 [`scoring.py`](../turnitover/verifier/scoring.py) 生成指标。旧20物体/60案例已归入 `data/benchmarks/uniphys-verifier-dev/`，只作开发回归。**正式 Test 尚未执行真实模型评测。**

**已实现：正式 Test 身份与未见边界。** [`select_uniphys_test.py`](../scripts/select_uniphys_test.py) 排除 `UPB_00000000–UPB_00000019`、旧 Test 全部对象和现有 Train/Validation，再按固定 seed、来源身份、完整模型 SHA 与规范化对象名称筛选20个对象；选择过程不渲染或人工查看候选。冻结清单位于 [`uniphys-test.json`](../configs/uniphys-test.json)，[`convert_uniphys.py`](../scripts/convert_uniphys.py) 默认从该清单生成 `data/assets/uniphys-test-catalog/`，记录 revision、输入哈希、许可证、continuous joint 策略和唯一 `split_group`。程序化审计确认正式 Test 与开发、Train、Validation 的对象交集均为0，三类近重复信号交集也均为0。

**已实现：UniPhys Part 2 Episode 工程验证划分。** [`select_uniphys_trainval.py`](../scripts/select_uniphys_trainval.py) 按固定 seed 执行 URDF 条件检查和完整 Mesh 转换预检，收集内部仍标作100个 Train 与20个 Validation 的对象；这两个名称只用于验证 Episode 生成、隔离与加载链路，冻结配置标记为 `episode_engineering_validation`、`formal_training=false`，**不作为正式 SFT 训练数据**。正式 Test 随后以这120个对象为排除集合，保证对象无交叉。现有工程产物含300个 Train case、60个 Validation case和360份逐案例预检；[`run_uniphys_verifier_batch.py`](../scripts/run_uniphys_verifier_batch.py) 可复用冻结身份检查，但结果只用于工程验证。

**已实现：4资产无模型 Episode Smoke 与监督质量修复。** [`generate_uniphys_supervision.yaml`](../configs/generate_uniphys_supervision.yaml) 复用既有生成器与独立 checker；真实执行得到12个样本（clean / 单缺陷 / 组合缺陷为3/6/3），12个均通过编译和逐缺陷隔离 checker，0 quarantine、0生成失败；prepared 为9 Train / 3 Validation。初版84行 Episode 错把3个 clean 的隐藏“无注入标签”当成 `pass` 证据，且12/12都机械执行满6步。修复后 [`episodes.py`](../turnitover/oracle/episodes.py) 不再由私有标签空集产生 `pass`：3个 clean 与1个未暴露 offset 均在预算耗尽后输出 `uncertain`；8个有可见 finding 的样本在首个证据步骤终止，得到51行 Episode、8 fail / 4 uncertain，终止原因为8 `evidence_sufficient` / 4 `budget_exhausted`，步骤分布为1步3个、2步3个、3步2个、6步4个。所有 finding 的身份和 `evidence_steps` 均与实际公开历史前缀一致，未暴露缺陷不进入目标。

[`audit_episode_export.py`](../scripts/audit_episode_export.py) 还按图像 SHA、公开 context 与历史对下一动作分组：修复版39个动作目标形成25个精确可见前缀组，0组存在不同动作，但15组为单例；Oracle 明确使用私有 counterfactual exposure，因此“未发现精确冲突”不能证明目标可由模型输入识别，动作监督仍是特权教师且可学习性未解决。修复前后产物为 `output/uniphys40k-episode-quality-before.json`、`output/uniphys40k-episode-quality-after.json`，新版 Episode 位于 `data/benchmarks/uniphys40k-oracle-smoke-v2/`。这是监督链路 Smoke，**不是正式训练集或模型能力结果**。

**停止目标现已显式区分。** `first_failure` 面向快速拒绝，在首个有证据 finding 后停止；12样本导出51行，8个 fail 在1–3步停止，组合缺陷允许 `partial`。`comprehensive_within_budget` 面向预算内尽可能完整诊断，保留完整轨迹并只输出所有已暴露 finding；导出84行，7 complete / 1 partial / 4 none，但不声称找全隐藏缺陷。目标名称同时写入 manifest、每行元数据和模型可见 payload，审计拒绝缺失或混用；两份实跑产物分别位于 `data/benchmarks/uniphys40k-oracle-first-failure-v3/` 与 `data/benchmarks/uniphys40k-oracle-comprehensive-v3/`。

**已完成：5 资产最小 SFT 与独立 Validation 对照。** 正式 catalog 现在逐资产保存并校验 `formal_training`、上游 revision/source ID、源 ZIP SHA、转换 spec SHA；prepared、Episode loader 和评分链路继续核对候选与参考程序身份及图像 SHA。15 个真实样本全部通过独立 checker，按原始资产隔离为 Train 3资产/9样本、Validation 2资产/6样本；62 行 Episode 的15个 terminal 目标为9 `fail` / 6 `uncertain`、0 `pass`。Qwen3-VL-2B LoRA 只训练9个 terminal 诊断/停止样本，不使用特权动作，seed `20261013`、3 epochs、27步，loss 由2.418降至0.261；adapter SHA-256 为 `c9ddf61f86cf40f208933e0aee6c8abe930435a52d7ecf43b262c9ce32a69bb7`。

同一6样本 Validation、同一图像/历史、`max_new_tokens=512` 下，底座→SFT 的合法终止输出为5/6→6/6，status accuracy 均为3/6；exact verdict 为3/6→0/6，finding 均为0 TP，SFT 另产生6 FP。SFT 在3个有证据 fail 上都停止并判 fail，但把3个应为 uncertain 的预算耗尽案例全部误报为 fail，说明小数据训练产生了明显 fail/offset 偏置。它只验证 checkpoint 可训练、可加载、可受控比较，**不支持泛化、pass 判断或主动取证能力主张**。

复现命令与主要产物：

```bash
python scripts/convert_uniphys40k.py --root data/uniphys40k-pilot --out data/assets/uniphys40k-trainval-v3 --ids UP_00011006,UP_00015016,UP_00015020,UP_00039007,UP_00039025 --motion-reviews configs/uniphys40k-trainval-v2.json
python -m turnitover generate --config configs/generate_uniphys40k_sft_smoke.yaml
python -m turnitover prepare-dataset --shards data/benchmarks/uniphys40k-sft-smoke-v1/raw/shard-0000.tar --splits data/assets/uniphys40k-trainval-v3/splits.json --out data/benchmarks/uniphys40k-sft-smoke-v1/prepared
python scripts/run_oracle_supervision.py --benchmark data/benchmarks/uniphys40k-sft-smoke-v1 --config configs/generate_uniphys40k_sft_smoke.yaml --out data/benchmarks/uniphys40k-sft-oracle-v1 --budget 6 --episode-objective first_failure
CUDA_VISIBLE_DEVICES=1 .venv-qwen/bin/python scripts/train_qwen_episode_sft.py --dataset data/benchmarks/uniphys40k-sft-oracle-v1 --model models/Qwen3-VL-2B-Instruct --out output/uniphys40k-qwen3vl2b-terminal-sft-v1 --seed 20261013 --epochs 3
CUDA_VISIBLE_DEVICES=1 .venv-qwen/bin/python scripts/evaluate_qwen_episode_sft.py --dataset data/benchmarks/uniphys40k-sft-oracle-v1 --model models/Qwen3-VL-2B-Instruct --out output/uniphys40k-qwen3vl2b-base-val-v1
CUDA_VISIBLE_DEVICES=1 .venv-qwen/bin/python scripts/evaluate_qwen_episode_sft.py --dataset data/benchmarks/uniphys40k-sft-oracle-v1 --model models/Qwen3-VL-2B-Instruct --adapter output/uniphys40k-qwen3vl2b-terminal-sft-v1/checkpoint --out output/uniphys40k-qwen3vl2b-sft-val-v1
python scripts/compare_qwen_sft.py --base output/uniphys40k-qwen3vl2b-base-val-v1/metrics.json --sft output/uniphys40k-qwen3vl2b-sft-val-v1/metrics.json --out output/uniphys40k-qwen3vl2b-controlled-comparison-v1.json
```

关键产物为 `data/benchmarks/uniphys40k-sft-oracle-v1/manifest.json`、`output/uniphys40k-sft-data-audit-v1.json`、`output/uniphys40k-sft-processor-validation-v1.json`、`output/uniphys40k-qwen3vl2b-terminal-sft-v1/{run.json,loss.jsonl,train.log.jsonl,checkpoint/}`、两份逐案例 `predictions.jsonl` 和 `output/uniphys40k-qwen3vl2b-controlled-comparison-v1.json`；均为运行产物，不进入代码提交。

**单参考 `pass` 仅在显式受限 scope 下可审计。** [`pass_evidence.py`](../turnitover/oracle/pass_evidence.py) 只读取模型可见的 task、参考图、观察历史与 runtime payload：`front_rest_appearance_only` 仅在关节动作前取得 front 图且与唯一 front/rest 参考逐像素一致时通过；`runtime_budget_only` 仅在公开、有限、非负的 triangle/draw-call 数值均不超公开阈值时通过。二者都明确排除关节、遮挡面和其他未观察属性，不能解释为整个 articulated object 正确。当前12条的 task 是广义 articulated verification，故可通过数为0；虽然7/12的 front/rest 完全一致，事后质量分析显示其中只有3个 clean，另有3个 joint-axis 和1个未暴露 offset，证明单图相等不足以支持广义 `pass`。12/12也没有 runtime 查询。因此3个 clean 继续为 `uncertain`，规则不读取 `labels=[]`、私有程序或 Checker 结果。

**配对参考图是广义运动判断的必要独立实验条件，但不是充分条件。** 现有每资产固定 witness grid 为16状态，6步轨迹只到达5–6个唯一状态；为当前12条轨迹逐观察加入同状态 reference，终态图像输入会从7张增至13张（+86%），全套 Episode 行累计图像槽位从336增至588（+75%），需要72个配对槽位、按资产缓存后27张唯一 reference。若只把预渲染 reference 作为输入，观察动作预算仍为6；若配对也计入动作预算，则最多只能比较3个状态。现有 Oracle 已为4资产私下渲染64张完整 reference grid（总 teacher render 448），但未按状态公开给模型；即使导出，6步仍覆盖不了16状态。因此下一实验应固定较小的必查 scope 或单独提高预算，并公开 state key、成功状态和同状态图像对；缺失配对、覆盖未完成或预算耗尽一律 `uncertain`，不得把私有全网格审计当作模型证据。

实际成功命令如下；同参数直接运行 `python scripts/run_oracle_supervision.py --benchmark data/benchmarks/uniphys40k-episode-smoke-v1 --config configs/generate_uniphys_supervision.yaml --out data/benchmarks/uniphys40k-oracle-smoke-v1` 曾因当前环境未 editable-install 而在导入阶段失败，改用模块入口后成功，未产生失败样本：

```bash
python -m turnitover generate --config configs/generate_uniphys_supervision.yaml --n-samples 12 --out data/benchmarks/uniphys40k-episode-smoke-v1/raw
python -m turnitover prepare-dataset --shards data/benchmarks/uniphys40k-episode-smoke-v1/raw/shard-0000.tar --splits data/assets/uniphys40k-trainval-v1/splits.json --out data/benchmarks/uniphys40k-episode-smoke-v1/prepared
python -m scripts.run_oracle_supervision --benchmark data/benchmarks/uniphys40k-episode-smoke-v1 --config configs/generate_uniphys_supervision.yaml --out data/benchmarks/uniphys40k-oracle-smoke-v1
python -m scripts.audit_episode_export --benchmark data/benchmarks/uniphys40k-episode-smoke-v1 --oracle data/benchmarks/uniphys40k-oracle-smoke-v1 --catalog data/assets/uniphys40k-trainval-v1/catalog.json --out output/uniphys40k-episode-smoke-v1-audit.json
python -m scripts.run_oracle_supervision --benchmark data/benchmarks/uniphys40k-episode-smoke-v1 --config configs/generate_uniphys_supervision.yaml --out data/benchmarks/uniphys40k-oracle-smoke-v2
python -m scripts.audit_episode_export --benchmark data/benchmarks/uniphys40k-episode-smoke-v1 --oracle data/benchmarks/uniphys40k-oracle-smoke-v2 --catalog data/assets/uniphys40k-trainval-v1/catalog.json --baseline output/uniphys40k-episode-quality-before.json --out output/uniphys40k-episode-quality-after.json
python -m scripts.run_oracle_supervision --benchmark data/benchmarks/uniphys40k-episode-smoke-v1 --config configs/generate_uniphys_supervision.yaml --out data/benchmarks/uniphys40k-oracle-first-failure-v3 --episode-objective first_failure
python -m scripts.run_oracle_supervision --benchmark data/benchmarks/uniphys40k-episode-smoke-v1 --config configs/generate_uniphys_supervision.yaml --out data/benchmarks/uniphys40k-oracle-comprehensive-v3 --episode-objective comprehensive_within_budget
python -m scripts.audit_episode_export --benchmark data/benchmarks/uniphys40k-episode-smoke-v1 --oracle data/benchmarks/uniphys40k-oracle-first-failure-v3 --catalog data/assets/uniphys40k-trainval-v1/catalog.json --out output/uniphys40k-first-failure-v3-audit.json
python -m scripts.audit_episode_export --benchmark data/benchmarks/uniphys40k-episode-smoke-v1 --oracle data/benchmarks/uniphys40k-oracle-comprehensive-v3 --catalog data/assets/uniphys40k-trainval-v1/catalog.json --out output/uniphys40k-comprehensive-v3-audit.json
```

**可复用的既有基础。** ReplicaCAD 数据引擎已支持按原始资产分组、腐蚀样本和私有真值隔离；Oracle 动作/成对诊断导出、Qwen3-VL 处理器适配及 assistant-only loss 见 [`dataset-pipeline.md`](dataset-pipeline.md) 和 [`oracle-supervision.md`](oracle-supervision.md)。正式 PartNet terminal Episode 的普通/证据约束 SFT 已完成；可见信息动作教师仍未完成。

**已实现：PartNet-Mobility 正式 Train/Validation 增量。** 官方批准后固定上游 revision，从160个确定性抽样 ZIP 中得到94个结构候选；48个优先候选经真实浏览器运动门禁与人工三视图/端点复核后35个通过（72.9%），形成28 Train / 7 Validation，覆盖21类、42 revolute / 18 prismatic、15个多关节物体。140条 corruption 样本中138条通过独立 checker、2条隔离；对应 Episode 已完整导出并通过逐像素重渲染复核。来源、许可、拒绝原因、隔离策略、命令和扩量成本见 [`training-dataset-expansion.md`](training-dataset-expansion.md)。自然生成错误仍为0且单独登记，未用注入标签冒充人工 Gold；数据建设阶段本身未启动训练，后续既有 SFT 记录见下文，冻结 Test 未修改。

**已完成：SFT 监督语义修复与第二轮 v4 重训。** v4 复用全部观察、Gold 和像素，修复138条 terminal 解释并增加539条逐前缀停止目标。普通/证据两组已用相同底座、539个 Train 样本、seed、LoRA、loss 权重和1617步预算重训；429条停止监督完全相同，只有110条诊断目标按完整 annotation / 当前可见证据区分。两份新 checkpoint、曲线、配置和审计均已保存，旧 v3 只作历史参考。

## 关键缺口

**第二轮 v4 诊断收益有限，不支持更准确的视觉利用。** 7个独立 Validation 物体、28条 terminal Episode 上，普通/证据组 exact 均17/28，finding 与定位 F1 均.421，TP/FP/FN 均8/11/11；证据组仅把 unsupported 15→11、defect-type F1 .737→.842。两组 status 都28/28，但无图像预算基线同样28/28，故不能归因于视觉；缺少合法的阶段匹配图像反事实与多 seed。

**逐前缀停止格式已学会，但停止时机未改善。** 普通/证据组 stop accuracy 为84/110、80/110，均低于无图像预算基线91/110；首次停止都只有10/28 exact、18/28提前，证据组提前 stop 前缀更多（27 vs 22）。step 1 的18个无证据前缀两组全误停，step 2 证据组又对9个无证据前缀全停，不能解释为依据证据充分性。预算耗尽9/9 `uncertain` 虽正确，但与阶段捷径一致。

**UniPhys-40K Train/Validation 资产池已做一次有界扩充，但覆盖仍很小。** 官方 `UniPhysGen@3f1bb2d` 代码确认发布 OBJ 为 `qpos=0`、轴/枢轴使用 OBJ 坐标、旋转范围为弧度、滑动范围为 OBJ 单位；[`convert_uniphys40k.py`](../scripts/convert_uniphys40k.py) 据此逐资产推导尺度，按三角形坐标多重集验证完整模型闭合，并核验来源许可及正式 Test 的来源/OBJ SHA/名称重叠。本轮先提交 Episode 修复 `66e8eb6`，再从此前未覆盖的6个分片各轮询10个、下载60个 ZIP；本地总计243个 ZIP，经轻量检查得到64候选。完整预检为8 structural candidate / 56拒绝：尺度无法验证26、关节轴字段不完整21、运动依赖重叠6、Objaverse 上游许可无法统一确认3。

对上述失败逐例复核后，26个尺度失败在部件/整物体 AABB 与 OBB 下均无一满足8%三轴一致性，允许移动部件描述也没有新增证据；最接近的两个整物体 AABB 仍为8.40%和8.33%，故未放宽阈值。21个字段不完整资产都缺少 `axis/pos/range` 中至少一项；其中3个缺失组虽被完整组包含，但这不足以证明它是重复标注而非额外未知自由度，21个均保守拒绝。6个依赖案例中仅 `UP_00003008` 与 `UP_00023014` 是参数完整的层状依赖；转换器现只支持“任意两组互不相交或严格包含”的层状树，并把子关节锚点转到父坐标系，同时要求 revolute 枢轴由 owner 与父侧刚性部件共同支持。两者随后仍因独立尺度门禁失败，故修复前后均为8 structural candidate、5 eligible，Train 3 / Validation 2，关节4 prismatic / 1 revolute；64候选与冻结 Test 三种去重信号仍为0重叠。逐资产数值、21项分类、6项结论及备选来源评估见 [`uniphys40k-v2-failure-audit.json`](../configs/uniphys40k-v2-failure-audit.json)。本轮重新运行8个 structural candidate 的真实浏览器运动检查为7机械成功/1失败；成功项的程序及三视图渲染 SHA 与此前审核逐一相同，因此物理结论仍为5通过/3拒绝。

三视图真实浏览器运动审核后，新增 `UP_00011006`（3D-FUTURE removable-lid prismatic）通过；`UP_00035009` 的折叠椅腿在发布正向终点穿过座椅，`UP_00000015` 的第二关节在 opposite 视图完全不可见，均拒绝且未猜测轴符号或放宽门槛。v2 catalog 因此为5资产、3 Train / 2 Validation，来源为 HSSD 2 / PartNet 2 / 3D-FUTURE 1，类别3种，关节为4 prismatic / 1 revolute；所有原始对象和 split group 唯一，冻结 Test 的来源、完整 OBJ SHA 和规范化名称重叠均为0。可复现审核输入见 [`uniphys40k-trainval-v2.json`](../configs/uniphys40k-trainval-v2.json)，产物位于 `data/assets/uniphys40k-trainval-v2/`。3D-FUTURE 只许可已接受协议者用于科研，禁止商业化及外部服务；HSSD 与 PartNet 也保留各自非商业条款，UniPhys 注释的 CC BY 4.0 不覆盖上游网格。这仍是小规模、研究用途受限的资产池，不是正式训练集或模型结果。

本轮下载与最终 catalog 命令如下；`ids` 来自第二条命令生成的64条实际候选，而非手工补名单：

```bash
python data/uniphys40k-pilot/find_uniphys40k_articulated.py --root data/uniphys40k-pilot --download --max-download 60 --stop-after 10000 --max-parts 20 --groups 4 12 20 28 36 41 --output output/uniphys40k-new-download-candidates-v1.jsonl
python data/uniphys40k-pilot/find_uniphys40k_articulated.py --root data/uniphys40k-pilot --local-only --stop-after 10000 --max-parts 20 --output output/uniphys40k-expanded-candidates-v3.jsonl
ids=$(python -c "import json; from pathlib import Path; print(','.join(json.loads(x)['asset_id'] for x in Path('output/uniphys40k-expanded-candidates-v3.jsonl').read_text().splitlines()))")
python scripts/convert_uniphys40k.py --root data/uniphys40k-pilot --out data/assets/uniphys40k-trainval-v2 --ids "$ids" --motion-reviews configs/uniphys40k-trainval-v2.json
python -m scripts.convert_uniphys40k --root data/uniphys40k-pilot --out output/uniphys40k-expanded-conversion-v6 --ids "$(jq -r '.assets[].asset_id' output/uniphys40k-expanded-conversion-v4/conversion.json | paste -sd, -)" --test-plan configs/uniphys-test.json --test-root data/assets/uniphys/data
for asset in $(jq -r '.assets[] | select(.status=="structural_candidate") | .asset_id' output/uniphys40k-expanded-conversion-v5/conversion.json); do python -m scripts.render_uniphys40k_smoke --catalog output/uniphys40k-expanded-conversion-v5/catalog.json --asset-id "$asset" --out "output/uniphys40k-expanded-smoke-v5/$asset"; done
python -m scripts.convert_uniphys40k --root data/uniphys40k-pilot --out output/uniphys40k-trainval-v2-revalidated --ids "$(jq -r '.assets[].asset_id' output/uniphys40k-expanded-conversion-v4/conversion.json | paste -sd, -)" --test-plan configs/uniphys-test.json --test-root data/assets/uniphys/data --motion-reviews configs/uniphys40k-trainval-v2.json
```

**v5 证据驱动停止专项已完成训练与评测（2026-10-15）。** Qwen3-VL-2B stopping-only LoRA实际训练1,618步；7物体/389行Validation上准确率91.26%、stop precision/recall 98.83%/89.08%，提前停止3、漏停31，严格同预算视觉对照正确84/102。全图遮蔽后准确率降至44.73%、主动stop召回降至0、配对正确降至0/102，而预算强制stop仍69/69，支持模型实际利用像素而非只看预算。结果仅为单seed、八参考姿态下的程序化形状一致性停止，不扩展为关节诊断；完整多任务第三轮仍未启动。详见[训练数据记录](training-dataset-expansion.md)。

## 下一步（按优先级）

1. **P0｜验证v5停止监督的可迁移性。** 在不改冻结Test的前提下补多seed和更多独立物体；下一轮完整多任务SFT将v5作为受控停止目标并保留diagnosis replay，继续分层报告主动/强制终止。有公开证据前继续禁止clean `pass`。
2. **P0｜按实测漏斗继续扩充 PartNet-Mobility。** 优先补 Train 混合关节、未覆盖类别和更大的独立 Validation；达到80–100个合格原物体前持续保存许可、来源/内容 SHA、运动审核及冻结 Test 去重证据。
3. **P0｜消除特权动作教师依赖。** 动作选择不读取私有 label/exposure，或显式学习可观测条件下的动作分布；完成前不把 terminal SFT 解释为主动取证训练。
4. **P1｜收集自然错误独立人工 Gold。** 固定 Generator，由不读取注入标签的人工依据真实渲染标注；GAPartNet 仅在取得完整官方包和额外许可后接入。
5. **P1｜冻结方案后执行60-case Test 与归因。** Test 不参与调参；统一模型、预算、策略和 seed，并按原始物体报告诊断、取证、停止及修复收益。

优先跑通**小规模且可信的完整链路**，不要在训练前追求覆盖全部 29 种缺陷或扩大到不可审计的数据规模。

---

最近同步：**2026-10-15 · v5视觉停止专项SFT**；训练、四组对照、严格配对和图像遮蔽评测已完成，支持将停止监督受控并入下一轮多任务SFT，但尚无多seed或关节诊断结论。完整多任务训练未启动；冻结Part 2 Test未读取或修改。
