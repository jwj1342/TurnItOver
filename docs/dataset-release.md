# Easy 与 Medium 候选发布

公开代码仓库与真实数据分别版本化。公开 Git 仅保存构建、验收、离线审核模板和小型固定契约；真实 RGB 候选与审核 ZIP 存放于私有仓库 [mmk1209/TIO-datasets](https://github.com/mmk1209/TIO-datasets)。`data/`、`output/`、`benchmark/easy/` 和 `benchmark/medium/` 不随公开代码提交。

| 候选 | 当前版本 | 物体 / 案例 | 数据划分 |
| --- | --- | ---: | --- |
| [Easy](easy-release.md) | `easy-candidate-v0.1.0-rc.2` | 127 / 275 | Train 117 / 251；Validation 10 / 24 |
| [Medium](medium-release.md) | `medium-candidate-v0.1.0-rc.1` | 75 / 150 | 全部为既有 Train 来源候选；未新增 Validation |

两者有 19 个 Easy Train 来源身份重用，合并后为 183 个不同来源身份；Medium 与 Easy Validation 的来源身份交集为零。物体数不能直接相加作为独立评测覆盖。Medium 不是独立于 Easy 的留出评测，后续划分仍须保持来源身份分组。

## 当前审核状态

- Easy：275 条历史 ledger 中，242 待审、32 完成、1 拒绝；新增 150 条均保持待审。
- Medium：两轮真实难度观察共 16 条，其中 11 条 Medium、2 条 clean、3 条 too_obvious。后五个物体已排除；当前 64 条缺陷尚无人审通过记录，150 条完整诊断模板均待填。
- 两者均为候选数据，不是冻结 benchmark；审核空白项、构造标签、自动校验和助手观察不等同独立人工结论。

Easy rc.2 保留 rc.1 的两个 ZIP 内容与 SHA，只修复私有外层发布清单与公开代码契约不一致的问题。旧版本及源研究工作树不覆盖。

## 访问与验收

合作者提供 GitHub 用户名、项目和用途，由私有仓库所有者邀请访问；沿用现有 PartNet-Mobility/ShapeNet 条款。私有访问不授予公开再分发许可。完整候选含 Gold 和历史判断，只交给协调者；审核者只接收对应的去答案审核 ZIP。

```sh
git fetch origin laq/dataset-release
git switch --detach FETCH_HEAD
git clone https://github.com/mmk1209/TIO-datasets.git data/TIO-datasets
python scripts/validate_easy_release.py \
  --manifest benchmark/releases/easy-candidate-v0.1.0-rc.2.json \
  --assets data/TIO-datasets/releases/easy-candidate-v0.1.0-rc.2 \
  --output data/verified/easy-rc2
python scripts/validate_medium_release.py \
  --manifest benchmark/releases/medium-candidate-v0.1.0-rc.1.json \
  --assets data/TIO-datasets/releases/medium-candidate-v0.1.0-rc.1 \
  --output data/verified/medium-rc1
```

每个 `--output` 必须不存在。验收以代码固定清单为信任入口，逐项核对附件、SHA、解压文件库存、来源绑定、审核隔离和候选完整性；所有检查通过后才发布最终提取目录。普通 `python -m pytest` 只使用合成 fixture，不下载数据，不要求存在真实源目录。

## 构造与复现边界

真实图片直接复制已有渲染结果，没有模型生成、改图或重新选择相机。完整构造程序、mesh/URDF、历史版本与排除案例媒体继续由原保管者在本地保存，不进入候选附件。包内 registry 保留可校对的哈希与历史来源引用；下载包的便携验收不冒充重放完整几何检查，也不提供程序训练导出。

详细构建、独立审核与回件流程分别见 [Easy](easy-release.md) 和 [Medium](medium-release.md)。
