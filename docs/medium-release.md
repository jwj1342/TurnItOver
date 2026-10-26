# Medium 候选数据发布

当前版本为 `medium-candidate-v0.1.0-rc.1`，以已封存的本地 `benchmark/medium` 为源。来源清单 SHA-256 为 `e0a930f82ff1e9fb86c7b2feaf5de47ac7ede1180980c9c5185fd3b757979071`，基于未冻结、含未提交生成工作的研究快照；代码基线和来源字节分别记录，不把基线 commit 冒充包含全部生成数据的提交。

公开代码中的固定契约为 [medium-candidate-v0.1.0-rc.1.json](../benchmark/releases/medium-candidate-v0.1.0-rc.1.json)。实际 RGB 附件存放于私有仓库 `mmk1209/TIO-datasets/releases/medium-candidate-v0.1.0-rc.1/`，不在公开 Git 或公开 Release 中上传图片。访问流程见 [总览](dataset-release.md)。

## 数据与规则

75 个独立物体、150 条 Clean/真实缺陷配对样本：轴错误 45、方向反转 22、部件偏移 8。全部沿用 Train 来源身份，未从当前候选重新制造 Validation 划分。与 Easy Train 重用 19 个来源身份，与 Easy Validation 无来源身份交集。

基础合同为 `medium-strength-v3`。收紧选择策略 `medium-selection-tightening-v1` 将旋转轴偏差限制为不超过 20°、平移轴偏差不超过 30°，并检查实际原生投影差异及错误运动相对正常运动的幅度。弱轴补采使用旋转轴 15°/20°、平移轴 15°/20°/30°；位移仍为实际物体包围盒对角线的 1%–3%。相机由正常对象确定并冻结，最终难度结合实际图像筛选。有限人审反馈校准不等于独立验证。

两轮 16 条真实难度反馈原文、时间、来源与字节前缀哈希均保留：11 Medium、2 clean、3 too_obvious。五个人工否决物体未回流。当前 11 条缺陷有 Medium 观察，64 条待确认；150 条完整定位诊断仍 pending，训练导出为空。未将原观察转换为新独立审核结论。

## 四个私有附件

- `medium-candidate-v0.1.0-rc.1-data.zip`：解压为 `medium/`，包含当前有效样本的 900 张 native 三态图、600 张输入帧、150 张对比板、150 张预览，以及 Gold、checker、原始人审与来源证据。协调者使用，不作为独立盲审输入。
- `medium-candidate-v0.1.0-rc.1-review.zip`：解压为 `medium_review/`，只含 150 条匿名案例的 900 张原生图片、离线审核页面和空白回件；原图仅重命名，没有修改像素。
- `release-manifest.json`：与公开代码固定契约逐字节相同，包含两个 ZIP 的大小、SHA、库存和预期完整验收报告。
- `SHA256SUMS`：两个 ZIP 和发布清单的校验和。

审核包不含来源 case/object ID、Gold、历史判断或匿名对应表。对应表只在完整候选 `private/review_mapping.jsonl` 中，由协调者保管。两包物理分开不构成权限强制隔离；审核者在返回独立判断前应只取得审核包。

完整构造程序、原始 mesh/URDF、历史版本和排除案例媒体继续 `local_custody`。`construction_registry.json` 仅记录原程序路径、大小和哈希；Gold 中的 program/source 路径是原始来源元数据，便携验收不会在下载者机器上解引用。包内 `source/automatic-checks.json` 是已封存源数据的几何校验记录，当前发布验收检查其字节绑定，不宣称重新渲染或重放源几何。

## 下载后完整验收

使用 `laq/dataset-release` 分支或对应 PR 的具体代码提交，不依赖尚未发布的代码 tag：

```sh
git fetch origin laq/dataset-release
git switch --detach FETCH_HEAD
git clone https://github.com/mmk1209/TIO-datasets.git data/TIO-datasets
python scripts/validate_medium_release.py \
  --manifest benchmark/releases/medium-candidate-v0.1.0-rc.1.json \
  --assets data/TIO-datasets/releases/medium-candidate-v0.1.0-rc.1 \
  --output data/verified/medium-rc1
```

输出目录必须不存在；错 SHA、缺图、额外文件、人审或资格被改、审核包不同源图片、匿名对应关系不完整均失败，失败不发布部分提取结果。通过后得到 `medium/` 与 `medium_review/` 两个目录，不需要本机 `output/` 或原构造源目录。

## 离线审核与回件

审核者解压独立审核 ZIP 后打开 `medium_review/review.html`，填写本人标识，观察三态播放与逐帧对比，再选择难度和说明。页面默认空白，存储按包与审核者隔离，需要本人显式载入保存记录；结果只在浏览器本地，不自动上传。下载的 JSONL 含匿名 ID、reviewer、decision、notes、reviewed_at 五字段。

```sh
python scripts/prepare_medium_review.py validate --root /path/to/medium_review
python scripts/prepare_medium_review.py validate-return \
  --root /path/to/medium_review --return-file /path/to/real-return.jsonl
```

回件校验只检查格式与案例集合，不认证审核者身份、不合并历史判断、不自动授予训练资格。协调者取得真实回件后，使用完整候选中的 private 映射对齐源案例，保留原文及分歧；本次尚无这些匿名案例的新独立回件。

## 从便携快照重新打包

已完整验收的候选可以重建附件，不需要原始构造程序。输出目录必须全新；相同 Python/zlib 环境下 ZIP 成员排序、时间与权限固定，重复构建应逐字节一致：

```sh
python scripts/build_medium_release.py \
  --root data/verified/medium-rc1/medium --output output/medium-rebuild \
  --tag medium-candidate-v0.1.0-rc.1 \
  --base-code-commit 7d6928eae65ce0c745c4b6a13baf92370460d14c \
  --code-ref refs/heads/laq/dataset-release
```

原始保管者可用 `build_medium_snapshot.py --source <封存源目录> --output <全新副本> --namespace medium-candidate-v0.1.0-rc.1 --source-code-commit <真实基线40位SHA>` 导出候选；该步骤验证源库存并按有效 case 白名单复制，不更改源数据。任何内容变化都使用新候选版本和新固定契约，不覆盖已发布附件。
