# Table 2 / Table 3：v1.1 复现代码

对应 [Issue #10 的结果报告](https://github.com/jwj1342/TurnItOver/issues/10#issuecomment-6905295193)。基础训练／评测提交为 `89e07db82890af02ae651dcd8b86082981fb7d1b`。数据必须是 `three-tier-900-sft-v1.1-pre.1`，manifest SHA-256：

`b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b`

本目录只提供代码、协议与校验值。私有数据、标签、案例程序、图片、checkpoint 和浏览器 profile 不包含在提交中。以下命令中的路径由运行者指定；复现输出应放在独立的 `sessions/` 目录或个人 scratch，不能提交公开仓库。

## 文件与改动边界

| 文件 | 用途 |
| --- | --- |
| `table2/evaluate_table2.py` | 调用固定评测器，投影观察条件并输出 findings F1 |
| `table2/table2_projection.py` | 保留指定观察，重映射证据步骤，投影目标 finding |
| `table2/visibility.py` | 从冻结曝光标注复算 Table 2 Visibility |
| `table2/summarize_table2.py` | 审核 21 组逐例预测、重算五类 F1、汇总三个 seed |
| `prepare_table3.py` | 校验本地私有输入，在新目录准备 Table 3 |
| `table3/run_table3.py` | 渲染观察网格，运行策略，实际重放不同轨迹 |
| `table3/planner.py` | 带显式回零动作的原实验局部 planner |
| `table3/browser.py` | 本地 Chromium/CDP；支持显式浏览器路径 |
| `table3/summarize.py` | 来源物体 bootstrap、CSV/LaTeX/报告 |
| `table3/audit_results.py` | 从图像独立复算曝光，并重建全部策略路径 |
| `table3/test_analysis.py` | 无浏览器的动作成本、合法路径、物体 bootstrap 检查 |
| `provenance.json` | 原始脚本哈希、运行代码及渲染依赖校验值 |

Table 2 的评测／投影脚本，以及 Table 3 的执行、planner、汇总、审计脚本，均逐字节保留原实验版本。`browser.py` 只改浏览器路径选择和 Windows 启动标志的兼容处理；`visibility.py` 只把本地固定路径改成参数。准备入口和 Table 2 汇总入口是新增的工程包装。没有修改基础训练框架、评分代码、动作预算或像素阈值。

## 环境

优先复用已兼容的训练和渲染环境。仓库标准 Python 环境见根目录 `requirements.txt`，Qwen 依赖见 `requirements-qwen.txt`；本次 Table 3 原始运行使用 Python 3.11、Chromium 140 / Playwright Chromium build 1187、ANGLE SwiftShader。表格代码不调用外部模型 API，不需要 API 密钥。

Table 3 额外使用 `websocket-client`（Python 导入名 `websocket`），并使用仓库的 NumPy、Pillow。指定已安装的 Chromium/Chrome：

```bash
export CHROMIUM_EXECUTABLE=/path/to/chromium
```

Windows PowerShell 使用 `$env:CHROMIUM_EXECUTABLE = 'C:\path\to\chrome.exe'`。未设置时，Windows 优先检测原 Playwright build 1187 的安装位置，否则从 PATH 查找 Chromium/Chrome。不同浏览器或渲染平台的像素结果可能不同；运行目录会记录浏览器和 GL 信息，不承诺跨平台逐字节一致。不要用 `python -O`，冻结实验的校验使用了断言。

GPU 评测和受管服务器上的批量渲染遵守该环境调度要求。下列命令应在已分配的计算资源中运行，不直接在登录节点启动批量任务。

## Table 2

数据输入为完整解压的 v1.1 数据目录，包含 manifest、私有标注、Episode 导出及图像。模型输入为本地底座权重和可选 LoRA checkpoint；不自动下载权重，也不选择 checkpoint。Ordinary / Evidence 使用相同底座，各自独立训练，seed 为 `20261013`、`20261113`、`20261213`。不能复用 v1.0 checkpoint 冒充 v1.1。

从仓库根目录设置：

```bash
CODE="$PWD"
PKG="$CODE/sessions/table23-reproduction"
DATA=/path/to/three-tier-900-sft-v1.1
MODEL=/path/to/Qwen3-VL-4B-Instruct
CHECKPOINTS=/path/to/v11-training-runs
RUNS=/path/to/new-table2-runs
mkdir -p "$RUNS"
```

`CHECKPOINTS` 需要有 `ordinary-<seed>/checkpoint` 和 `evidence-<seed>/checkpoint`。2B、4B、8B 使用各自底座及匹配的 adapter，其他参数相同。完成一个规模的 21 组评测：

```bash
for key in base ordinary-20261013 evidence-20261013 ordinary-20261113 evidence-20261113 ordinary-20261213 evidence-20261213; do
  adapter=()
  if [ "$key" != base ]; then adapter=(--adapter "$CHECKPOINTS/$key/checkpoint"); fi
  python -X utf8 "$CODE/scripts/evaluate_qwen_episode_sft.py" \
    --dataset "$DATA" --dataset-manifest-sha256 b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b \
    --model "$MODEL" --out "$RUNS/full-$key" --task diagnosis --split validation \
    --device cuda --max-new-tokens 512 --bootstrap-samples 1000 "${adapter[@]}"
  for condition in viewpoint motion; do
    python -X utf8 "$PKG/table2/evaluate_table2.py" \
      --runtime "$CODE" --dataset "$DATA" --model "$MODEL" \
      --out "$RUNS/$condition-$key" --condition "$condition" "${adapter[@]}"
  done
done
python -X utf8 "$PKG/table2/summarize_table2.py" --dataset "$DATA" \
  --runs "$RUNS" --out "$RUNS/summary.json"
python -X utf8 "$PKG/table2/visibility.py" --runtime "$CODE" --dataset "$DATA" \
  --out "$RUNS/visibility"
```

每个评测输出目录必须不存在；不覆盖已完成结果。上面的完整观察入口保留原实验的 1,000 次来源物体 bootstrap。`evaluate_table2.py --condition full` 也可计算相同 F1，但该包装没有请求 bootstrap，不能拿它补造原始置信区间。汇总中的三 seed 均值和最小／最大值不是置信区间。

固定评分口径：

- Motion/Full 使用 104 例（60 clean、44 defective；Easy/Medium/Hard 为 39/35/30）。Viewpoint 只有 Hard 30 例（20 clean、10 defective）；Easy/Medium 写 N/E。
- Visibility 仅以相应条件支持的 defective 为分母：Viewpoint 4/10，Motion 43/44，Full 44/44。隔离案例不算“未暴露”。
- 五类固定 Macro-F1，零正例类别按 0 计。所有模型使用相同条件的 Evidence 目标；无效输出保留并计漏检。All 从逐例结果计算。
- 缩短观察重映射证据引用，只保留在该子集中有证据的 findings，不沿用完整输入 pass 认证。子集的 `internal-evaluator-metrics.json` 仅作审计，不发布其中的状态／pass 指标；使用 `table2-f1.json`。
- Viewpoint 的 All 是 Hard-only，不能和另两行的 104 例 All 直接比较。跨条件比较用共同 Hard 子集，并说明观察条件会改变可见目标。
- 曝光为冻结像素代理，复合案例沿用原曝光归因，不声称逐 finding 人工 grounding。

## Table 3

需要两份已有私有输入：

1. v1.1 数据目录（至少 manifest、`annotations.private.jsonl`、`validation.episodes.jsonl`）。
2. 对应的 `three-tier-900-validation-repair-58-v1` 程序包，含 `cases.json`、`SHA256SUMS`、`views.json`、`runtime/`、`cases/`。这是参考／候选源码与编译程序的载体；最终准入和单缺陷选择仍由 v1.1 标注决定，不能用程序包原来的 58 例直接跑 Table 3。程序包需由已有权限的合作者从私有渠道取得，本分支不包含它。

准备入口从 v1.1 选择全部 34 个准入单缺陷 Validation 案例，核对参考／候选身份、TS/JS 校验值、视角文件及渲染 runtime。输入应为已核验、可信的发布附件。固定渲染文件哈希见 `provenance.json`，不要用不明版本的重编译文件替换。

```bash
WORK=/path/to/new-table3-work
python -X utf8 "$PKG/prepare_table3.py" --dataset "$DATA" \
  --repair-bundle /path/to/three-tier-900-validation-repair-58-v1 \
  --runtime "$CODE" --out "$WORK"
cd "$WORK"
python -X utf8 test_analysis.py
python -X utf8 run_table3.py
python -X utf8 summarize.py
python -X utf8 audit_results.py
```

`WORK` 必须是新目录。输出含私有案例、图像和逐例结果；准备入口写入忽略全部内容的 `.gitignore`。仅用 `--limit 1` 可做单例预检，不能汇总为完整结果。脚本可跳过本目录已完成的轨迹，恢复时必须保持输入、脚本、浏览器和环境不变；更改任一项应重新准备独立目录。

协议：34 例／34 个来源物体，19 Easy、15 Medium；5 个 part_offset、12 个 joint_axis、17 个 direction_flip。44 个准入 defective 中的 10 个复合案例不纳入本实验，因为继承的曝光记录不支持逐缺陷归因。未按策略结果筛选。

网格为配置中的 12 个静态视角，并在 front / oblique_fr 展开各关节非零 lower/mid/upper，任一时刻只有一个关节偏离初始状态。256×256、双方共享参考取景；曝光条件是归一化 RGB MAE ≥ 0.001 且通道差异 > 0.02 的像素比例 ≥ 0.001。

预算 K=1/2/4/6。所有策略以 front 静止观察开始并计一次，所以 **K=1 三种策略完全相同**。换视角、驱动、恢复关节各计一次。`planner.py` 的局部适配对不在三档中的零位发出显式驱动回零动作，仍计一次；轨迹开始前初始化状态不提供免费观察，轨迹内不免费重置。

Random 为每例 seed 0–99；先在例内平均，再进行 1,000 次来源物体 bootstrap（seed 20261116，95% 百分位区间）。Greedy oracle 按新增曝光／动作成本选择，包含中间观察和恢复成本，不是全局最优或学习策略。First exposure 的未暴露案例记 K+1；单缺陷子集使 All recall 与 Coverage 相同。结果不等于诊断或修复成功率。

`summary.json`、`table3.csv`、`table3.tex` 是汇总；`final-audit.json` 记录独立图像／路径复核。正式核对需要完整 34 例，1,340 张网格图像、13,872 次策略运行、6,733 条不同轨迹、32,640 次实际重放动作。不能把相同轨迹共享的重放次数当作独立随机样本。

## 本次发布检查

见 [VALIDATION.md](VALIDATION.md)。检查结果区分重新执行、已有产物复算和未覆盖范围；不以本次代码打包冒充新模型实验。
