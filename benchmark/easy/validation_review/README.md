# Validation 独立双人盲审

本目录可单独离线分发：固定现有 Validation 的 **24 例、10 个物体**。审核输入只含 case ID 和媒体，不含构造标签、checker、历史审核结论或训练资格。两个空模板均未填写，不代表已经开展或完成审核。

1. `jwj1342` 解压后直接用浏览器打开 `jwj1342.html`；`WinterCCC` 打开 `WinterCCC.html`。无需服务器、网络或仓库其他文件。页面没有自动上传功能。
2. 各自独立完成全部24例；提交前不讨论个案、不查看对方结果。每项均可选“无法判断”，不要为填满表格而猜测。
3. 点击“下载本人结果 JSONL”，单独交回协调者。浏览器本地保存只是便利功能，请下载备份；可用“导入本人备份”恢复。直接编辑各自 `.blank.jsonl` 也可，但必须遵守下面的 schema。
4. **先分别交回，再合并和对齐分歧**。协调者只在双方完整结果都已收到后运行 merge；对齐产生的新记录另行保存，保留独立原件。脚本不会覆盖根 `human_review.jsonl`，也不会把一致意见自动变成验收通过。

页面上排 Reference、下排 Candidate，三列 rest / mid / limit。输入预览也是上排 Reference、下排 Candidate，左 rest、右 limit。旧 board 顶部含历史审核状态；本目录副本统一裁掉顶部60像素，保留完整六宫格与运动状态标注，原始图片不变。`media_provenance.json` 记录源相对路径、源 SHA-256、crop 坐标和副本 SHA-256；它不含答案。

## 返回格式

每行一个对象：`case_id`、本人 `reviewer`、`review_status`、`verdict`、`defect_type`、`one_glance`、`no_clipping_or_occlusion`、`notes`、`reviewed_at`。空模板 `review_status=pending`，判断及时间为 null。完整返回须为 `completed`，包含带时区 ISO 时间（页面使用 UTC），两个质量项均为 boolean。

- `verdict=normal` → `defect_type=none`。
- `verdict=defect` → `defect_type` 为 `direction` / `axis` / `offset` / `other`。
- `verdict=uncertain` → `defect_type=uncertain`；请在备注说明原因。

这些是审核者的判断选项，不是预先提供的答案。merge 拒绝空模板、未完成项、同名审核者、重复/缺失/额外 case ID、无效枚举和缺少时区的时间。输出 `report.json` 汇总一致/分歧与需跟进的案例，`disagreement.jsonl` 保留分歧字段及双方原始判断；备注差异本身不视作分类分歧。即使双方一致，无法判断或画面质量问题仍列入 `needs_followup`。

## 协调者命令（仓库根目录）

重新生成需要项目已有 Pillow；审核者离线打开页面不需任何依赖。prepare 只读取现有 split 和媒体映射，不读取 Gold 或历史审核记录。

```sh
python scripts/prepare_easy_validation_review.py prepare --zip /tmp/easy-validation-review.zip
python scripts/prepare_easy_validation_review.py validate
python scripts/prepare_easy_validation_review.py merge \
  --first /path/to/jwj1342.validation-return.jsonl \
  --second /path/to/WinterCCC.validation-return.jsonl \
  --output /tmp/easy-validation-adjudication
```

ZIP 在 `/tmp`，不加入 Git。只分发 ZIP 或本目录；不要将根 Gold、旧审核 ledger、双方返回文件或分歧报告放入分发目录。身份字段并非身份认证；协调者仍需从两位审核者分别取得真实独立回件。
