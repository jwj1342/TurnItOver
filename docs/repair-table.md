# 修复主表运行层（judge 条件、缓存、作业）

`scripts/run_repair_table.py` 在一组带程序的缺陷案例上，用同一个冻结生成器和同一个修复循环（`turnitover.repair.loop.run_repair`，精确文本编辑协议，`RepairEnvironment.audit` 独立审计）逐个 judge 条件跑完，输出可以直接填论文修复表的指标。judge 的实现在 `turnitover/repair/judges.py`。

## judge 条件

所有 judge 每轮给生成器的图完全相同：案例的整组观察，按格子顺序（先视角，再 rest、mid、limit）每格先参考图、后当前程序的候选图，取景固定。反馈里的 `observations` 逐张列出视角、状态、参考或候选和图名，`image_order` 给出顺序。judge 之间只差反馈里各自的那一节：

| judge_id | 反馈里 judge 自己的一节 | 用到的模型 |
| --- | --- | --- |
| `none` | 无 | 无 |
| `pixel` | `pixel_difference`：每格参考与候选的像素差是否超过阈值，不给缺陷类型和部件 | 无 |
| `zs8b_screenshot` | 零样本 Qwen3-VL-8B 看参考图加一张静止渲染给出的 verdict | 8B 基座 |
| `zs8b_sweep` | 同一个 8B 看整个固定扫描给出的 verdict | 8B 基座 |
| `ours_2b_ordinary` `ours_2b_evidence` `ours_8b_evidence` | 微调验证器看整个固定扫描给出的 verdict | 基座加 LoRA |
| `gold` | 由私有腐蚀记录和特权审计生成的 findings，上界；没有 `--labels` 时直接报错 | 无 |

verdict 类 judge 的一节是 verdict 本身（status、findings 含 part、defect_id、joint_id、severity、suggested_fix、evidence_steps）加 `evidence_step_cells`，把 verdict 引用的每个步号对应到观察格子（视角、状态、参考或候选）。步号按 judge 自己看到的图计，screenshot 和 sweep 的步号含义不同，所以要这张对照表。verdict 按方法节规则校验，finding 必须引用存在的步、fail 至少一个 finding、pass 不能有 finding，无效 verdict 变成 uncertain 并记录原因（`judge_call.json`）。

`pixel` 的阈值是 `scripts/exp900/pixel_status_baseline.py` 在 900 训练集上拟合的 0.0010002983617596328（记录在 `tio-runs/out/level900-base-2B4B-26612686/pixel-status/metrics.json`，同一个分数：一对参考与候选图 RGB 平均绝对差，归一到 [0,1]），定义在 `judges.py` 的 `PIXEL_THRESHOLD`。

## 固定扫描与提示词

扫描按案例元数据定义，视角乘以关节状态，状态是 rest、mid、limit。参考和候选用同一个固定取景（`Case.framing`）渲染，judge 看的图和生成器收到的图都是这些。judge 提示词里的图片顺序和文件命名与 three-tier-900 训练导出一致，每个视角先参考图后候选图，单视角命名 `NN-reference_rest.png`，多视角命名 `NN-left_candidate_mid.png`。提示词用 `turnitover/repair/prompt_900.txt`（她导出分支的 `verifier/prompt.txt`，比主分支多 joint_id 规则）加 JSON 载荷，键顺序 context、remaining、history、image_order、taxonomy_ids、training_objective。context 的 parts 和 joints 顺序取参考程序的 AssetSpec（`Case.parts`、`Case.joints`），与导出相同；浏览器加载程序得到的顺序在 58 例里有 48 例不同，不能用。`tests/unit/test_repair_judges.py` 用 `load_pack` 读出的真实案例对 58 例的训练记录逐字节核对（数据路径取 `$TIO_RUNS`，默认 `<repo>/tio-runs`）。

渲染按 (扫描, 取景, 程序哈希) 缓存在 `<out>/cache/renders/`，verdict 按 (模型身份, 提示词, 图片字节) 缓存在 `<out>/cache/verdicts/<judge>/`，模型身份含基座名、max_new_tokens 和 adapter 的 `adapter_config.json` 加 `adapter_model.safetensors` 内容哈希（作业会把 checkpoint 拷到节点本地，所以不用路径）。生成器回复按 (生成器, 提示词, 图片哈希) 缓存在 `<out>/cache/generator/`。已完成的 episode（`result.json` 的 status 为 complete）会跳过，可以重复提交续跑。

## 案例目录格式

每个案例一个目录，和 `scripts/prepare_repair_experiment.py` 的输出一致。

```
<case>/initial.ts             待修复的候选程序
<case>/reference.private.ts   参考程序，只用于审计和渲染参考图
<case>/case.private.json      {"framing": {"center":[x,y,z],"radius":r} 或 null,
                               "tier": "easy|medium|hard",
                               "sweep": {"views": ["front"], "joints": ["joint_0"], "states": ["rest","mid","limit"]},
                               "annotation": {"id": "...", "labels": [{"defect_id": "...", "parts": ["..."], "severity": 0.5, "params": {"joint": "..."}}],
                                              "grid": [...], "exposures": [...]}}
```

`tier` 和 `sweep` 可选。没有 `sweep` 时视角取 `--default-views`，关节取标签 params.joint 的并集（没有则取程序全部关节），状态按层次，Easy 是 rest 和 limit，其余三态。`grid` 和 `exposures` 可选，有的话 gold judge 只引用曝光该缺陷的格子。合作者导出 900 例的候选程序后，按这个格式放进目录即可，`sweep.views` 按导出记录的 context.views 填。

## 运行

登录节点只跑无模型的 judge 和脚本生成器的冒烟。正式实验走 `scripts/slurm/repair_table.sbatch`（在仓库根目录提交，账户和 GPU 在命令行给，完整命令见 `docs/vulcan-handoff.md`），环境变量 CASES、OUT、JUDGES、GENERATOR（api 或 scripted-stop、scripted-fix），api 时再给 GENERATOR_MODEL，例如 `moonshotai/kimi-k3` 或 `anthropic/claude-fable-5.1`，模型 id 会覆盖 `.env` 里的 `TIO_GENERATOR_MODEL`，凭证仍从 `.env` 读。生成器调用不设 temperature，客户端没有这个参数，论文要写不能逐次复现。

```bash
CASES=... OUT=... JUDGES="none pixel zs8b_screenshot zs8b_sweep ours_2b_ordinary ours_2b_evidence ours_8b_evidence gold" \
GENERATOR=api ROUNDS=3 sbatch --account=<账户> --gpus-per-node=<GPU>:1 scripts/slurm/repair_table.sbatch
python scripts/summarize_repair_table.py --out <OUT> [<OUT2> ...] --cases <CASES> [--case-ids-file eligible.txt]
```

汇总表的列是成功率、按缺陷类型的 resolved（该标签涉及部件的全部审计不变量在最终程序上通过）、定位率（finding 的部件与同类型标签部件相交）、到首次通过的生成器调用数和 token、非改进率、状态重访率和供应商报告费用。

## 她的 58 例修复包

私有仓库 release `three-tier-900-validation-repair-58-v1`（`mmk1209/TIO-datasets`），解压在 `/scratch/jwj/tio-runs/data900/repair58/three-tier-900-validation-repair-58-v1/`，runner 对 `--cases` 指向这个目录时按 `manifest.json` 的 `format` 字段自动识别。

```
manifest.json      format three-tier-900-validation-repair-programs-v1，58 例，Easy 20 Medium 19 Hard 19，只有 validation
cases.json         列表，每项 case_id、difficulty、data_identity、exposure、target_joint_id、
                   state_values {rest, mid, limit}、observation_cells [{view_id, state, reference_image, candidate_image}]、
                   views、programs {candidate|reference: {typescript, javascript, spec, program_sha, typescript_sha256}}、image_sha256
cases/<id>/        candidate.ts reference.ts（Program ABI v1，几何内嵌）、同名 .mjs、.spec.json（AssetSpec）
images/<id>/       原始观察图，命名 NN-reference_rest.png 或 NN-left_candidate_mid.png
runtime/ views.*   three r170、harness、program_abi.ts（与仓库一致）、views.yaml（与仓库一致）
SHA256SUMS verify.mjs verification.json browser-verification.json
```

包里没有缺陷标签，runner 用 `--labels` 从 900 导出的 `annotations.private.jsonl` 按 case id 取 labels，用 `--episodes` 从 `validation.episodes.jsonl` 的证据目标推出每个格子曝光哪些标签，gold judge 据此引用步骤。扫描按 `views`、`target_joint_id` 和 `observation_cells` 的状态定义，关节值直接用 `state_values`（她的 rest 是 0，limit 是上限，mid 是她记录的值，18 个目标关节下限不为 0，所以不能用档位推算）。加载时校验 typescript_sha256。

包里没有记录取景，她的图也不是自动取景（自动取景下物体偏大，例如 eb-0f542119f7b0 宽 108 像素，原图 86 像素）。取景由 `scripts/exp900/derive_pack_framing.py` 恢复并写进 `turnitover/repair/framing_900.json`，`load_pack` 读它填 `Case.framing`。规则：参考程序在目标关节 rest、mid、limit 三个值下（其他关节为 0）全部顶点的世界坐标包围盒取并集，中心是盒中心，半径是半对角线乘按层次的余量，Easy 1.25、Medium 1.5、Hard 1.15（三层由三个不同的发布构建器渲染）。`scripts/exp900/check_pack_framing.py` 用修复循环的同一条渲染路径重画第 0 轮的参考图和候选图，与原图逐张比较。2026-11-15 在每层 3 例（共 66 张）上核对：52 张逐字节相同，其余 14 张各只有 1 到 3 个轮廓像素不同（单像素最大绝对差 94/255，整图平均绝对差不超过 0.001），物体包围盒全部一致。剩下的差异来自构建器的浮点细节（用双精度包围盒能消掉其中一部分，但不是全部），包里没有足够信息复现。其余 49 例只按同一规则推导，没有逐张渲染核对，可在 CPU 作业里对全部 58 例运行该脚本。

## 当前状态（2026-11-11）

- 单元测试、登录节点脚本修复器检查和她的 58 例修复包加载都已通过；GPU 路径（基座加 LoRA 的 judge）尚未在作业里验证，作业因队列原因取消。
- 生成器通过 `.env` 配置。目前可用的只有经中转站访问的 Kimi K3（Anthropic 消息格式，模型名 `k3`，带 thinking 块），该中转站在 Cloudflare 后面，`turnitover/models/client.py` 已加 User-Agent 头。Claude 侧尚无通道。
- 费用估算：她的程序约 70 KB，一次调用两万多 token 输入；58 例、8 种 judge、3 轮、两个生成器约 2800 次调用。
- 作业脚本已移到 `scripts/slurm/repair_table.sbatch`，按 judge 分作业并行，用 `scripts/exp900/eligible_cases.py` 定资格集，计算节点无外网时用 `scripts/exp900/connect_proxy.py` 转发。11-15 的执行计划见 `docs/vulcan-handoff.md`。
