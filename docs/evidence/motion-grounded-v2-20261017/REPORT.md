# Motion-grounded SFT v2：2026-10-17 完成报告

显式运动差异中间监督**没有带来可靠的未见物体缺陷诊断改善**。两组实际完成 GPU 训练；移除输入中的数值差异特征后，冻结 B 协议 offset 仍为 0/4，clean 也为 0/4。停止重复 Qwen3-VL-2B LoRA，不接入主动观测或 v5 停止策略。

## 训练与输入

先在 `laq/sft-training` 提交本地改动 `327b138`，随后运行本轮实验。离线基座 `../TurnItOver/models/Qwen3-VL-2B-Instruct`，同一 RTX 4090 D 型号，直接组 GPU0、中间组 GPU1；未终止其他 GPU 任务。

复用 Train 来源物体的既有 Chromium 渲染：18 个训练物体、108 行；主验证为 5 个训练未见物体、30 行。每个视角将原始 256×256 图排成 512×512 对照，列为参考/候选，行为 rest/actuated；rest-only 只有一行。四个视角为 front/oblique_fl/back/right。输入保留合法 hierarchy、被测试 joint、公开状态及面板 step 对应；不含私有几何、注入参数、Gold 或数值 RGB 差异特征。轴/偏移/干净反事实共享物体、视角、状态；证据不足行有 uncertain。

两组的公开提示词、图像字节、行顺序及诊断 target 完全相同。中间组仅在 assistant target 前增加 `static_rest/static_steps/motion_change/motion_pairs`。这些中间标签来自固定 RGB 测量，是像素比较代理，不是人工标注的三维旋转语义。训练 admission 依赖像素可见性/签名，验证集没有按成功结果筛选；故不能把主验证分数称为独立语义 Gold 准确率。

每组 108 样本 × 2 epochs = **216 个真实 optimizer steps**；seed=20261016、AdamW lr=2e-4、LoRA rank8/alpha16/dropout0、q/k/v/o 投影、bf16、SDPA、batch1、梯度裁剪1、assistant-only mask。步数、样本、优化器预算相同；中间组 target 更长，监督 token 数/FLOPs 不相等。单 seed，无显著性或 bitwise 复现主张。

## 主要结果

下列两列依次为普通直接分类 / 增加中间监督；完整正确同时要求分类、joint/part 和受支持的合法证据。分类统计与严格 schema/ID/step 验证分开保留，当前新六组输出均通过严格验证。

| 评测 | 直接分类 | 中间监督 |
|---|---:|---:|
| 主验证充分运动：分类正确（像素可见性代理） | 5/15 | 6/15 |
| 主验证充分运动：完整正确 | 4/15 | 2/15 |
| 主验证充分运动：offset 正确（可见 target） | 0/4 | 0/4 |
| 主验证充分运动：clean 正确 | 0/5 | 1/5 |
| 主验证充分运动：类别塌缩的同物体三元组 | 5/5 | 4/5 |
| 冻结 B：分类 / 完整正确 | 4/12；4/12 | 4/12；4/12 |
| 冻结 B：axis / offset / clean 召回 | 4/4；0/4；0/4 | 4/4；0/4；0/4 |
| 冻结 B：类别塌缩三元组 | 4/4 | 3/4 |
| 冻结 B：缺陷 part 正确（包含分类错误） | 5/8 | 4/8 |
| 冻结 B：被测试 joint 一致性 | 4/4 | 4/4 |
| 冻结 B：类型及引用支持 / 所有预测缺陷 | 4/12 | 4/10 |
| 冻结 B：类型、定位、引用全部支持 / 所有预测缺陷 | 4/12 | 4/10 |
| 冻结 C：应 abstain 的 axis/clean | 8/8 | 8/8 |
| 冻结 C：仍可见的 offset 正确 | 0/4 | 0/4 |

冻结 B 为四物体各 axis/offset/clean；直接组 12 例全判 axis。中间组 axis→axis 4；offset→axis 3、uncertain 1；clean→axis 3、uncertain 1。因此塌缩计数减少只是两次改判 uncertain，没有恢复 offset/clean 诊断。C 两组均全部 uncertain：缺证 abstention 成功，同时拒绝了 4 个有静态偏移证据的案例，不能称为校准成功。公开输入已给出被测试 joint，joint 一致性不是开放定位。

跨协议输入使用 front/oblique_fl、rest/mid/end（每视角三行面板），并保留同样的公开层级与模型接口；与四视角两状态训练协议不同。冻结四物体与主验证重叠，是复用的开发验证队列，不是新鲜或冻结 Test。额外两个训练未见物体仅有 2 clean + 6 uncertain（含 rest-only），无有效 axis/offset 阳性覆盖；其充分运动分类为 0/4→1/4、rest-only 均4/4，不能补足独立 offset 验证。

## 视觉使用对照与旧 v2 复核

去掉主验证图像但保留完全相同文字，直接组分类标签 **30/30 不变**，完整诊断四字段则27/30不变；中间组分类与完整诊断均25/30不变。充分运动分类分别5/15和5/15，完整正确5/15和2/15。中间监督的1例主验证分类增量没有变为完整诊断增量，更没有跨协议 offset 收益。这不证明所有输出都不依赖图像，但不支持学会运动比较的主张。

本轮还对已有的“输入提供 RGB 数值差异”的 v2 checkpoint 原始预测统一重评分，并未重训这些旧模型。主验证全30行分类直接27/30、中间26/30，完整20/30→22/30；冻结 B 分类9/12→8/12、offset 3/4→0/4、完整6/12→8/12。数值特征可提升部分分类，显式文本中间监督本身并未可靠改善未见物体缺陷区分。旧数值输入与新像素面板还存在表示/提示差异，因此跨两轮差值不是“仅删去一个特征”的严格因果消融；本轮两组之间才是严格匹配比较。

## 审计、失败保留与边界

独立审查采用 `experiment-audit` 技能的 fresh same-family agent；语义审计为 provisional，见 `AUDIT.md/json`。审查员独立复核全部536行评分一致，但在最终汇总时遇到用量限制，未交付最终独立签字；已保留初审、消息核验记录及明确标记的执行者收尾补充。最终10组全部由同一 scorer SHA 重算；逐例原始输出和独立 scored 行留在本地 output。评分修正：拒绝多余/不存在/重复/非整数 steps、非法 hierarchy ID；冻结 offset 使用原始不可变 `pixel_signals.exposed`，而不是新 erosion 阈值。旧派生 suite 有4行 offset 的证据 target 为空，不应据其旧 full 指标判断模型；现已在独立评分中恢复原始证据规则，未改原始冻结文件。`unsupported_finding` 包含分类、定位及引用失败，不是纯引用错误率。

最初面板文件名只绑定端点，造成部分 full/rest 面板覆盖；加载器拒绝后保留失败 `direct-data/intermediate-data` 和 `direct-train.log`，该次无训练步。改为绑定完整 tile SHA 序列后，通过552个图像引用/每组的哈希检查与逐 tile 原图核对，才运行216步。正式产物是 `direct-panels/intermediate-panels`、`direct-run2/intermediate-run`。

生成时 builder 字节版本已从编辑差异恢复，并与 manifest 记录的 SHA 精确匹配，见 `provenance/` 三份builder；恢复快照不是声称当时已归档。按文档实际执行prepare-only，616行JSON及全部图像哈希一致。派生 transfer manifest 继承了旧 `rows:146`，实际 `validation_rows:32` 与JSONL均为32；此过时字段不改写，保持推理绑定的 manifest 哈希，当前通用 builder 已修正未来派生产物。原始推理metrics中的带前缀full_motion计数也不作为最终分母；最终采用逐组重评分，通用evaluator已修复未来计数。

无付费 API、无 v5 数据/停止任务混合、无冻结 Test 读写；冻结 B/C 原始 inputs/Gold 与其预存 manifest SHA 完全一致。没有操作 dataset 分支或工作树，没有 push。

## 本地产物与复现

根目录：`output/motion-grounded-v2-20261017/`。

- Checkpoint：`direct-run2/checkpoint/`（adapter SHA `c8dd52d3d29a802262f4bae0b30d8d0a32b01c0fc815f7369de19aee53c47155`）、`intermediate-run/checkpoint/`（`a3b1a4804596f5168430db48dd49c0cf64b91152500d23b6965a434639ceb8d8`）。
- 两组 run.json、216步 loss.jsonl；六组 `*-eval/{predictions.jsonl,metrics.json}`，共184次本轮真实 GPU 生成。
- `final-scores/*/{scored.jsonl,metrics.json}`、`comparison.json`、`artifact-manifest.json`：10组历史/本轮结果统一评分、原始文件哈希、checkpoint与训练数据绑定。
- 仓库中保留本报告、comparison、artifact manifest、两组 run.json、审计、测试日志及精确 builder 快照；大图像和checkpoint在上述本地 output 路径，未进 Git，也未上传。

从仓库根目录执行（现有渲染数据及本地基座为前提；目标必须为新目录）：

```bash
# 仅准备输入，供独立复现核对；不会训练
MOTION_PREPARE_ONLY=1 bash docs/evidence/motion-grounded-v2-20261017/reproduce.sh output/FRESH_PREPARE_DIR
# 完整训练和三组评测命令归档；本轮结论后不再重复执行
MOTION_GPU=0 bash docs/evidence/motion-grounded-v2-20261017/reproduce.sh output/FRESH_RUN_DIR
# 已执行的针对性检查：14 passed
PYTHONPATH=. ../TurnItOver/.venv-local/bin/python -m pytest tests/unit/test_motion_grounding.py tests/unit/test_motion_diagnosis.py tests/unit/test_motion_pixel_control.py -q
```

## 下一步（方案，尚未执行）

停止2B LoRA的同类重试。先在新的 Train 来源物体上预先锁定三元组与可见性审核，不动冻结 Test，并保留可见 offset 阳性和真正缺证阴性；不能继续用已经开发过的4物体估计泛化。

第一条路线是离线更大视觉底座，例如官方 [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)，先做固定协议基线，确认其能读出细小位移和clean，再考虑一次有界适配。这是候选方案，不是已有收益或单卡可运行性承诺。

第二条路线是显式运动估计：仅从公开多视角RGB预测可动部件掩码及跨帧点轨迹，分别估计参考/候选运动，分离 rest 位移与运动方向/轨迹差异，输出残差与可观测性后再映射到合法 hierarchy。视角/遮挡不足时 abstain；不用私有几何或注入参数。应先用人工独立审核的轨迹/部件与相机一致性验证运动估计，再评价缺陷分类；单个像素残差不能等同旋转轴估计。两条路线都以未见物体 offset/clean 平衡表现、真实定位及受支持引用作为继续投入门槛。
