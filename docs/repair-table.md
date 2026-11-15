# 修复主表运行层（judge 条件、缓存、作业）

`scripts/run_repair_table.py` 在一组带程序的缺陷案例上，用同一个冻结生成器和同一个修复循环（`turnitover.repair.loop.run_repair`，精确文本编辑协议，`RepairEnvironment.audit` 独立审计）逐个 judge 条件跑完，输出可以直接填论文修复表的指标。judge 的实现在 `turnitover/repair/judges.py`。

## judge 条件

| judge_id | 生成器收到什么 | 用到的模型 |
| --- | --- | --- |
| `none` | 参考图和候选程序静止渲染各一张，没有 verdict | 无 |
| `pixel` | 每个扫描格子参考与候选的像素差是否超过阈值，不给缺陷类型和部件 | 无 |
| `zs8b_screenshot` | 零样本 Qwen3-VL-8B 看参考图加一张静止渲染给出的 verdict | 8B 基座 |
| `zs8b_sweep` | 同一个 8B 看整个固定扫描给出的 verdict | 8B 基座 |
| `ours_2b_ordinary` `ours_2b_evidence` `ours_8b_evidence` | 微调验证器看整个固定扫描给出的 verdict | 基座加 LoRA |
| `gold` | 由私有腐蚀记录和特权审计生成的 findings，上界 | 无 |

verdict 类 judge 的反馈是 verdict 本身（status、findings 含 part、defect_id、joint_id、severity、suggested_fix、evidence_steps）加它引用的观察图，第一张永远是参考图。verdict 按方法节规则校验，finding 必须引用存在的步、fail 至少一个 finding、pass 不能有 finding，无效 verdict 变成 uncertain 并记录原因（`judge_call.json`）。

## 固定扫描与提示词

扫描按案例元数据定义，视角乘以关节状态，状态是 rest、mid、limit。渲染顺序和文件命名与 three-tier-900 训练导出一致，每个视角先参考图后候选图，单视角命名 `NN-reference_rest.png`，多视角命名 `NN-left_candidate_mid.png`。提示词用 `turnitover/repair/prompt_900.txt`（她导出分支的 `verifier/prompt.txt`，比主分支多 joint_id 规则）加 JSON 载荷，键顺序 context、remaining、history、image_order、taxonomy_ids、training_objective，和训练记录逐字节一致（`tests/unit/test_repair_judges.py` 对真实导出记录做了核对）。

渲染按程序哈希缓存在 `<out>/cache/renders/`，verdict 按 (judge, 扫描, 程序哈希) 缓存在 `<out>/cache/verdicts/`，生成器回复按 (生成器, 提示词, 图片哈希) 缓存在 `<out>/cache/generator/`。已完成的 episode（`result.json` 的 status 为 complete）会跳过，可以重复提交续跑。只缓存正常结束的回复，截断的不缓存。生成器调用遇到 HTTP 429/500/502/503/504、连接失败或超时会重试，等待从 20 秒起翻倍、不短于 Retry-After、单次不超过 300 秒，每次调用总预算由 `TIO_GENERATOR_RETRY_BUDGET_S` 给（默认 1800 秒）。生成器最终失败（`ModelError`）只结束当前 episode，`result.json` 记 status `generator_error` 和错误信息，作业继续跑下一个案例，结束时以非零状态退出，重交会重跑这些 episode。

程序里不少于 200 个字符的数值数组字面量在给生成器看的文本里换成 `__GEOM_<k>__` 占位符（k 按出现顺序），生成器在省略后的文本上编辑，碰到、删除、复制或部分覆盖占位符的编辑按无效补丁处理，应用后占位符原样展开回原数组。58 例修复包上省略再展开逐字节一致，候选与参考的数组逐例相同，所以缺陷都在可见文本里。每轮在 `result.json` 的 rounds 里记 `program_chars` 和 `elided_program_chars`。

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
python scripts/summarize_repair_table.py --out <OUT> [<OUT2> ...] --cases <CASES> --case-ids-file eligible.txt
```

每个 judge 的分母都是 `--case-ids-file` 列出的案例（必填）。缺失、未完成和 `generator_error` 的 episode 在成功率里算失败，其标签算未解决，分别计入 Missing（缺失或未完成）和 Error 两列。汇总表的列是成功率及 Wilson 95% 置信区间、按缺陷类型的 resolved（该标签涉及部件的全部审计不变量在最后一次被采纳补丁的程序上通过，没有采纳过补丁时用初始程序）、定位率（finding 的部件与同类型标签部件相交）、Inv.（返回补丁的轮次中无效补丁的比例）、到首次通过的生成器调用数和 token、非改进率和状态重访率（只数补丁被采纳的轮次）以及供应商报告费用。后几列在完成的 episode 上按轮次或 finding 合并计算。

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

## 当前状态（2026-11-11）

- 单元测试、登录节点脚本修复器检查和她的 58 例修复包加载都已通过；GPU 路径（基座加 LoRA 的 judge）尚未在作业里验证，作业因队列原因取消。
- 生成器通过 `.env` 配置。目前可用的只有经中转站访问的 Kimi K3（Anthropic 消息格式，模型名 `k3`，带 thinking 块），该中转站在 Cloudflare 后面，`turnitover/models/client.py` 已加 User-Agent 头。Claude 侧尚无通道。
- 费用估算：她的程序约 70 KB，一次调用两万多 token 输入；58 例、8 种 judge、3 轮、两个生成器约 2800 次调用。
- 作业脚本已移到 `scripts/slurm/repair_table.sbatch`，按 judge 分作业并行，用 `scripts/exp900/eligible_cases.py` 定资格集，计算节点无外网时用 `scripts/exp900/connect_proxy.py` 转发。11-15 的执行计划见 `docs/vulcan-handoff.md`。
