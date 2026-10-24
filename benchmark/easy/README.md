# Easy 候选数据集

本目录是当前 Easy 候选集合：**127 个源物体、275 例**。其中125例来自 `easy-batch-v1/v2`，另150例来自明显缺陷导入批次，已获用户同意纳入集合。`collection_approval.json` 记录这次集合层面的决定，不代替逐例判断或独立盲审。

`splits.json` 固定物体隔离划分：Train 为117物体/251例，Validation 为10物体/24例。新增75物体/150例全部只在 Train。旧冻结 `benchmark/v1` 独立于本次交付。

当前仍是候选集合，尚未正式冻结或导出训练集。新增150例保持 `pending_human_review`；自动检查、抽查和集合纳入批准不等于逐例审核完成。全量历史审核 ledger 为242例待审、32例完成、1例拒绝。

## 文件与审核入口

- `private/gold.jsonl`：构造标签与来源信息，须与审核者输入隔离。
- `review_manifest.jsonl`：每例 board、输入预览和四张状态图的相对路径。
- `human_review.jsonl`：原逐例审核 ledger，记录历史审核状态。
- `review.html`：275例图像总览；历史图板可能含旧审核状态，不作为新的独立盲审入口。
- `validation_review/README.md`：24例 Validation 的独立双人盲审说明；`jwj1342.html`、`WinterCCC.html` 和各自空记录分别绑定本人。只向审核者分发这个独立目录，各自回件收齐后再比较。
- `private/import_checks.jsonl`：150例导入构造检查结果，是检查证据，不是人工批准。

## 来源与校验

`provenance.json` 记录源工作树快照、封存归档和本次转换。`archive/` 保存精简程序成员索引、原始 split 快照和恢复说明。新增例的 `program` 显式引用归档成员，而非声称媒体审核包中存在对应程序文件。完整构造归档目前仅有本地保管副本，尚未作为远端资产发布；程序重放须另行取得归档并核对 SHA-256。

`splits.json` 绑定当前相对 Gold 文件及实际哈希，历史源哈希保存在 provenance 中；原物体和案例分配保持不变。完整构造包与历史输入继续封存，本次不引入生成系统或历史 Medium 批次。

`bundle_manifest.json` 覆盖本目录除自身外全部文件的 SHA-256。从仓库根目录执行：

```sh
python scripts/validate_easy_bundle.py
python -m pytest tests/unit/test_easy_bundle.py tests/unit/test_easy_validation_review.py
```

校验覆盖文件清单、哈希、ID、划分保留、审核状态一致性、归档引用和独立盲审媒体来源。通过校验不代表人工审核、重新渲染、axis/sign 捷径消除或训练准备已完成。

## 后续工作

本次以 `Refs #9` 跟踪交付，不关闭 issue。axis/sign 表达等价随机化与重新审计、正常变化 Clean、Medium/Hard 构造、独立盲审回收、正式冻结和训练导出分别后续推进。捷径重审计通过前，code-reading 对比继续披露 Issue #8 记录的残余审计结果。训练脚本与旧冻结集合沿用各自既有版本。
