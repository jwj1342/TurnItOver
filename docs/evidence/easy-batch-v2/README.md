# easy-batch-v2：`joint_axis` 坐标轴置换批次（待抽查）

取代 `easy-batch-v1` 中的 47 个随机角度 `joint_axis` 样本（原因与做法见 `docs/code-shortcut-fix.md`）。缺陷 id 仍为 `kinematics.joint_axis`，`private/gold.jsonl` 的 `parameters.mode` 与 `stratum` 均为 `axis_permutation`：把参考轴 `±e_i` 换成另一条坐标轴 `±e_j`（j≠i），正负号由显式的 `numpy.random.Generator` 抽取。没有调用任何模型。

**人工审核（抽查 10 个）已完成：** 10 个置换样本全部判 fail，一眼可见，结论由用户逐例给出（`eb2-ac54fe009b34` 未指出具体错误）；其余 32 个仍为 `pending_human_review`。本批没有任何代填。未冻结 v2，没有 `FREEZE.json`，也未划分 Train/Validation。

| 项目 | 数量 |
|---|---:|
| 考察物体 | 50（easy-batch-v1 的全部物体） |
| 通过物体 / 样本 | 42 / 42，全部为 `axis_permutation` |
| 像素门槛下四个置换候选均未通过 | 8（其中 5 个在 v1 有随机角度样本） |
| 因参考轴不沿坐标轴而跳过 | 0 |

门槛、冻结相机和关节值与 `easy-batch-v1` 完全一致，参考图像与 v1 的 clean 对照逐张哈希核对（不一致会直接报错）。每个样本的 `clean_case_id` 指向 `easy-batch-v1` 中的 clean 对照，`supersedes` 指向被取代的 v1 样本。没有 v2 样本的 5 个物体（PM_100685、PM_10305、PM_10357、PM_11581、PM_9992）的处理见 `easy-batch-v1/README.md`。

## 抽查

规则与 v1 相同：`spot_check.json` 按固定哈希顺序取 10 个缺陷样本及其 clean 对照。`review.html` 盲审，**先不要打开 `private/`**；页面前 10 张是 v2 样本，其后是对应的 clean 对照（图板来自 `easy-batch-v1/boards`）。v2 样本的结论填在本目录的 `human_review.jsonl`；clean 对照的结论填在 `easy-batch-v1/human_review.jsonl`（10 个对照中 3 个已在第一次抽查中审过）。

## 已知问题

置换后的候选轴分布与 clean 轴分布不同（clean 以 x 轴为主，没有 `-z`；置换样本偏向 y/z 且含 `-z`），只读代码的统计规则仍能高于随机，见 `docs/code-shortcut-fix.md` 的审计数字与修改方案。这不是 #8 第 9 条的要求，残余推迟到后续层处理；本批用于 `code_read` 评测时须同时报告审计数字，不能描述为“代码层无捷径”。

## 复现与存放

候选 spec 放在 `data/benchmarks/easy-batch-v2-specs/`（不进 Git），`manifest.json` 记录 SHA-256。`python -m scripts.rebuild_easy_batch axis`，可中断后续跑。
