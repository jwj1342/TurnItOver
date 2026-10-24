# Easy 候选数据发布

代码与数据分别版本化：公开 Git 保存构建/验收脚本、盲审页面模板、固定 Validation ID/物体划分与小型发布契约；完整 RGB 数据包存放在私有仓库 [`mmk1209/TIO-datasets`](https://github.com/mmk1209/TIO-datasets)。公开仓库不提供 Easy 图片 Release 或公开下载链接。默认单测不下载数据、不读取真实图片，也不要求本地存在 `benchmark/easy/`。

## 当前候选版本

版本为 `easy-candidate-v0.1.0-rc.1`，来源为原 Draft PR #11 的提交 `27f7e03a42a3a835fbf9e2c09e04faf9e19695be`。本次重新整理交付方式，保留原案例、标签、划分、历史审核记录及原媒体；未重新采样或改变难度。

| 范围 | 物体 | 案例 |
| --- | ---: | ---: |
| Easy 候选全集 | 127 | 275 |
| Train | 117 | 251 |
| Validation | 10 | 24 |
| 新增明显缺陷导入批次（全部在 Train） | 75 | 150 |

新增150例获准纳入集合，仍为 `pending_human_review`；全量历史 ledger 为242例待审、32例完成、1例拒绝。当前不是冻结 benchmark 或训练导出。axis/sign 表达随机化及捷径重审计、正常变化 Clean、Medium/Hard 构造与独立审核回收继续按 Issue #9 分别推进；重审计前仍须披露既有 code-reading 残余审计。

## 私有数据包

- `easy-candidate-v0.1.0-rc.1-data.zip`：解压后为 `easy/`，包含275例媒体、Gold、checker证据、来源、划分和历史审核记录，供协调者与研究使用。
- `easy-candidate-v0.1.0-rc.1-validation-review.zip`：解压后为 `validation_review/`，包含24例独立盲审素材、两个离线页面与各自空白回填文件。
- `release-manifest.json`：与公开代码中的 `benchmark/releases/easy-candidate-v0.1.0-rc.1.json` 完全一致，绑定两个 ZIP 的大小、SHA-256、内部文件数量、解压大小与完整验收报告。
- `SHA256SUMS`：两个ZIP及发布清单的校验和。

完整包包含构造标签和历史结论，不能作为独立盲审输入。只有获得 `mmk1209/TIO-datasets` 私有仓库权限的协作者才能访问完整包；审核者只接收独立盲审 ZIP，并在提交独立判断前不查看完整包、旧 PR 答案或对方回件。

本次不包含原始 mesh/URDF、完整构造程序或封存的 `filesystem.tar.gz`。`archive/registry.json` 仍明确 `availability=local_custody`；程序重放需另行取得完整归档并检查其哈希。因此本包不是可独立重放或直接训练的程序数据集。来源包含 PartNet-Mobility，第三方来源的使用条款仍适用；见 [资产来源说明](asset-sources.md)。

## 上游条款与当前发布状态

候选图像由本项目基于 `sapien-sim/PartNetMobility` 渲染生成，官方 revision 为 `ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f`；对象身份与源归档哈希见包内来源映射。PartNet-Mobility 建立于 ShapeNet，官方条款规定数据库用于非商业研究与教育，并要求研究同事在访问前同意相关条款：[PartNet-Mobility 数据卡](https://huggingface.co/datasets/sapien-sim/PartNetMobility)、[ShapeNet 条款](https://huggingface.co/spaces/ShapeNet/README/raw/main/TERMS.md)。已核实文本没有明确说明本项目派生 PNG 的公开分发范围；代码许可也不构成上游数据库或派生图像权利的重新授权。

因此候选包已放入私有仓库，未在公开仓库创建或上传 Easy Release。公开派生 PNG 的授权范围仍待确认；私有仓库访问也不构成再分发许可。

## 申请访问

合作者应提供 GitHub 用户名、所属项目和用途，由 `mmk1209/TIO-datasets` 所有者按账号邀请为 private repository collaborator。获准后从 `releases/easy-candidate-v0.1.0-rc.1/` 下载并使用 `SHA256SUMS` 校验。PartNet-Mobility 与 ShapeNet 条款限制数据库用于非商业研究/教育，并要求研究同事访问前同意相关条款；不得将 RGB 文件转存到公开仓库、公开 Release、公共网盘或未经授权的第三方服务。

## 下载并完整验收

使用 Draft PR #12 的最新代码；在 PR 合并并创建新代码 tag 前，不要使用旧的公开 `easy-candidate-v0.1.0-rc.1` tag。以下命令从仓库根目录执行，下载目录只能包含本版本四个附件；GitHub 自动生成的 Source code ZIP 不是数据包。

```sh
git fetch origin pull/12/head:easy-private-data-code
git switch --detach easy-private-data-code
mkdir -p data/releases/easy-candidate-v0.1.0-rc.1/assets
git clone https://github.com/mmk1209/TIO-datasets.git data/TIO-datasets
cp data/TIO-datasets/releases/easy-candidate-v0.1.0-rc.1/* data/releases/easy-candidate-v0.1.0-rc.1/assets/
python scripts/validate_easy_release.py \
  --manifest benchmark/releases/easy-candidate-v0.1.0-rc.1.json \
  --assets data/releases/easy-candidate-v0.1.0-rc.1/assets \
  --output data/releases/easy-candidate-v0.1.0-rc.1/extracted
```

验收先核对代码内固定契约和附件哈希，再安全解压并执行完整275例校验；同时确认独立盲审包与候选快照中的审阅素材逐文件一致。缺文件、错哈希、错误划分或待验收目录冲突会失败，不跳过，也不覆盖已有提取目录。完整通过后才创建最终 `extracted/` 目录。

普通代码测试可直接执行 `python -m pytest`。完整发布验收必须单独执行上述命令，单测通过不代替真实附件验收。

## 独立盲审与回收

审核者只需下载独立盲审ZIP，解压后直接离线打开 `validation_review/jwj1342.html` 或 `validation_review/WinterCCC.html`，各自填写并下载本人JSONL，无需Python、服务器或完整数据包。空白模板尚未填写，不代表已有独立回件。

协调者收到双方独立回件后执行：

```sh
python scripts/prepare_easy_validation_review.py validate \
  --bundle /path/to/validation_review
python scripts/prepare_easy_validation_review.py merge \
  --bundle /path/to/validation_review \
  --first /path/to/jwj1342.validation-return.jsonl \
  --second /path/to/WinterCCC.validation-return.jsonl \
  --output /path/to/new-adjudication-directory
```

合并只输出分歧与需跟进项，不修改原回件、根审核ledger或训练资格。身份字段不构成身份认证，仍须分别取得真实回件。具体选项和schema见ZIP内README或 [模板说明](../scripts/easy_review_assets/README.md)。

## 重新构建

从已验收的候选快照生成新附件，显式提供数据目录；模板源位于 `scripts/easy_review_assets/`，不从下载数据中读取。以下输出目录必须不存在，且位于数据根之外：

```sh
python scripts/prepare_easy_validation_review.py prepare \
  --root /path/to/easy --output /path/to/new-validation-review
# 两个目标目录都必须不存在；仅更新独立候选副本，不改历史发布或源归档。
cp -a /path/to/easy /path/to/candidate-copy
mv /path/to/candidate-copy/validation_review /path/to/previous-validation-review
cp -a /path/to/new-validation-review /path/to/candidate-copy/validation_review
python scripts/validate_easy_bundle.py --root /path/to/candidate-copy --write-manifest
python scripts/build_easy_release.py \
  --root /path/to/candidate-copy --output /path/to/new-release-assets \
  --tag easy-candidate-v0.1.0-rc.1 \
  --source-commit 27f7e03a42a3a835fbf9e2c09e04faf9e19695be \
  --base-code-commit 7d6928eae65ce0c745c4b6a13baf92370460d14c
```

ZIP使用排序成员、固定时间和权限，不携带宿主文件时间。相同文件在相同Python/zlib环境下重复构建须产生相同哈希。要重现本版本，生成的清单必须与Git内固定契约逐字节相同；本地构建的清单本身不是新的信任依据。任何内容变更均使用新版本和新契约，不覆盖既有发布附件。
