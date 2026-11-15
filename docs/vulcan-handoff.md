# Vulcan 交接：截稿前的三层 900 例实验

写于 2026-11-15（EST），截稿 11-16 AoE，即 11-17 06:59 EST。

Nibi 从 11-15 08:10 起大面积停机，故障单 #1194，GPU 节点 39 个里 35 个 down。登录节点和 /scratch 还能用，所以实验改到 Vulcan 跑。

本文按执行顺序写，每一步都有检查点，检查点不过就不往下走。

## 要跑什么

论文主线用 `three-tier-900-sft-v1`（manifest sha256 `6d0fc534…`），不用 benchmark/v1，也不用 v2.1。

已经有的结果，Nibi 上 17 个 checkpoint 已训完、已评完：
- 暴露：只动关节 91.7%，只换视角 28.3%。
- 2B 的 Ordinary 对 Evidence：defect F1 8.9 对 24.6，Loc F1 10.6 对 32.0，引用支持 0 对 33.4。

还缺三项，按优先级排：

1. **修复主表（关键路径）。**
   - 数据是 58 例修复包。
   - 生成器只用 Kimi K3，也就是 `.env` 里的 `TIO_GENERATOR_MODEL=k3`，走 Anthropic 消息格式的中转站。
   - 跑 1 次重复，3 轮。
   - judge 有 6 个：`none` `pixel` `zs8b_sweep` `ours_2b_ordinary` `ours_2b_evidence` `gold`。
   - 时间够再加 `zs8b_screenshot` `ours_8b_evidence`。
2. **识别补全。**
   - 8B Ordinary 补 seed 20261213。
   - withheld 47 例，跑 2B 和 8B。
3. **可选。** 测试集 180 例只评 2B 和 8B。

不做：第二生成器、Claude judge、OpenHands、headroom 的 F1 列、oracle 观察选择表。

### 已知风险

- **Kimi K3 中转站不稳定。** 11-15 14:40 EST 从 Nibi 登录节点测了一次，先返回 HTTP 429，再次请求返回 503 "Service temporarily unavailable"。第 5 步的生成器冒烟是硬检查点，不通就按最后一节的 Plan B 换生成器。
- **代码基线。** 代码在分支 `exp/vulcan-900` 上，基于 `7d6928e`，没有合入 origin/main 上的 `e735f70`（11-04 的迭代合成功能），两者有 9 个文件重叠。截稿后再合并，现在不要在这个分支上 merge main。

## 0. 推荐做法：整个文件夹用 Globus 传过去

Nibi 上的 `/scratch/jwj/TurnItOver` 已经整合好了，模型、数据、checkpoint 和 `.env` 都按下面的布局放在里面。这些是硬链接，不额外占 Nibi 空间，Globus 会当普通文件传，没有符号链接。git 工作树停在分支 `exp/vulcan-900` 上。

- **传输**：把整个文件夹传到 Vulcan 的 scratch，共约 24GB，其中模型约 21GB。
- **加密**：Globus 传输选项里勾选 encrypt transfer，因为 `.env` 里有 API 凭证。
- **跳过的步骤**：这样传就不用做第 1 步的克隆，也不用做第 2 步的搬运和解压，只需要跑第 2 步最后的检查命令。

传完后在 Vulcan 登录节点先执行：

```bash
cd ~/scratch/TurnItOver               # Vulcan 上的实际路径
rm -rf .venv                          # Nibi 的 venv 在 Vulcan 不能用，第 3 步会重建
git status && git branch --show-current     # 应为干净的 exp/vulcan-900
git pull                              # 取我之后推送的修改
chmod 600 .env
```

`.cache/ms-playwright` 和 `web/node_modules` 留着也没关系，第 3 步的 bootstrap 会检查并补齐。

## 1. 克隆与目录布局（不整体传输时）

放在 scratch 上：

```bash
cd ~/scratch            # 或 Vulcan 上你的 scratch 路径
git clone https://github.com/jwj1342/TurnItOver.git
cd TurnItOver
git checkout exp/vulcan-900
```

git 里没有模型、数据、checkpoint 和 `.env`，全部按下表放进仓库目录。`models/` 和 `tio-runs/` 都已经在 `.gitignore` 里。

```
TurnItOver/
├── .env                          生成器凭证，单独传，chmod 600
├── models/
│   ├── Qwen3-VL-2B-Instruct/     4.0G  revision 89644892e4d85e24eaac8bacfd4f463576704203
│   ├── Qwen3-VL-8B-Instruct/     17G   revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
│   └── Qwen3-VL-4B-Instruct/     8.3G  revision ebb281ec70b05090aa6165b016eac8ec08e71b17（可选，本轮不用，整合的文件夹里没有）
└── tio-runs/
    ├── code900/                  她的训练评测脚本（laq/three-tier-900-sft-export 88ac0df），作业里 cd 进去运行
    ├── data900/
    │   ├── dl/                   900-private-candidates.zip（sha256 185d2d6b…）等
    │   ├── three-tier-900-sft-v1/episode_export/   由 zip 解压，修复 runner 从这里读标签和逐格暴露
    │   ├── three-tier-900-withheld-v1/             withheld 导出
    │   └── repair58/three-tier-900-validation-repair-58-v1/   58 例修复包
    ├── out/                      17 个 checkpoint 和评测结果（level900-*），新结果也写到这里
    ├── repair/                   修复表输出；calib-login16/ 是 Nibi 上 16 例校准的原始结果
    └── logs/                     作业日志（新建）
```

所有作业脚本都读两个环境变量，默认值就是上面的位置：
- `TIO_RUNS`，默认 `<仓库>/tio-runs`。
- `TIO_MODELS`，默认 `<仓库>/models`。

如果放在别处，提交作业前 export 这两个变量即可。

## 2. 用 Globus 从 Nibi 搬数据（不整体传输时）

整体传输的话跳过本节的搬运，只跑本节最后的检查命令，其中 unzip 那一行也可以跳过，因为已经解压好了。

Vulcan 的 collection 见 [Vulcan.md](https://docs.alliancecan.ca/wiki/Vulcan) 里的 Globus 链接，Nibi 用它自己的 collection。按下表搬，总量约 31GB，绝大部分是模型。

| Nibi 源路径 | Vulcan 目标 | 大小 | 用途 |
|---|---|---|---|
| `/scratch/jwj/tio-runs/models/Qwen3-VL-2B-Instruct/` | `models/Qwen3-VL-2B-Instruct/` | 4.0G | 2B judge、withheld |
| `/scratch/jwj/tio-runs/models/Qwen3-VL-8B-Instruct/` | `models/Qwen3-VL-8B-Instruct/` | 17G | zero-shot judge、8B 训练、withheld |
| `/scratch/jwj/tio-runs/code900/` | `tio-runs/code900/` | 3M | 训练评测脚本 |
| `/scratch/jwj/tio-runs/data900/dl/` | `tio-runs/data900/dl/` | 184M | 900 例数据 zip |
| `/scratch/jwj/tio-runs/data900/repair58/` | `tio-runs/data900/repair58/` | 823M | 修复包 |
| `/scratch/jwj/tio-runs/data900/three-tier-900-withheld-v1/` | `tio-runs/data900/three-tier-900-withheld-v1/` | 小 | withheld |
| `/scratch/jwj/tio-runs/out/` | `tio-runs/out/` | 827M | checkpoint 和已有结果 |
| `/scratch/jwj/TurnItOver/.env` | `.env` | 小 | 凭证，也可以 scp |

`code900` 也可以从 git 重建，不必搬：

```bash
mkdir -p tio-runs/code900 && git archive 88ac0df | tar -x -C tio-runs/code900
```

模型如果不想用 Globus，可以在 Vulcan 登录节点按固定 revision 从 Hugging Face 下载。下载前要先做完第 3 步的 bootstrap，因为它会建 `.venv`。

```bash
source scripts/setup_env.sh && pip install --no-index -q huggingface_hub==0.36.0
for spec in 2B:89644892e4d85e24eaac8bacfd4f463576704203 8B:0c351dd01ed87e9c1b53cbc748cba10e6187ff3b; do
  size=${spec%%:*}; rev=${spec#*:}
  hf download Qwen/Qwen3-VL-$size-Instruct --revision $rev --local-dir models/Qwen3-VL-$size-Instruct \
     --exclude '*.md' && echo $rev > models/Qwen3-VL-$size-Instruct/REVISION.txt
done
```

搬完后在登录节点检查：

```bash
chmod 600 .env
echo "185d2d6ba7ca2b6d2fb7426d407e50c9682a8632e3a827f18502cab9c309e2e0  tio-runs/data900/dl/900-private-candidates.zip" | sha256sum -c -
(cd tio-runs/data900 && unzip -q dl/900-private-candidates.zip)      # 得到 three-tier-900-sft-v1/episode_export
ls tio-runs/data900/three-tier-900-sft-v1/episode_export/{annotations.private.jsonl,validation.episodes.jsonl,test.episodes.jsonl}
ls -d tio-runs/out/level900-sft-*/*/seed*/*-diagnosis/checkpoint | wc -l     # 应为 17
ls models/Qwen3-VL-2B-Instruct/model.safetensors models/Qwen3-VL-8B-Instruct/model-0000{1..4}-of-00004.safetensors
python3 -c "import json;print(len(json.load(open('tio-runs/data900/repair58/three-tier-900-validation-repair-58-v1/cases.json'))))"   # 58
mkdir -p tio-runs/logs tio-runs/repair
```

## 3. 环境：不要搬 venv，在 Vulcan 重建

Nibi 上的 `.venv` 和作业里的 venv 不要用 Globus 搬过来，原因有三：
- venv 里写死了 Nibi 的绝对路径。
- 它链接的是 Nibi 上加载的模块。
- Playwright 浏览器和 `web/node_modules` 也和机器有关。

要搬的只有 `.env`，也就是 API 凭证，它不是 Python 环境。

重建分两层，都只从 Alliance wheelhouse 安装：
- **登录节点的 `.venv`**：给冒烟、单测和渲染用，由 `scripts/bootstrap_login.sh` 一次性建好。它同时下载 Playwright 的 chromium-headless-shell 到 `.cache/ms-playwright`，并执行 `npm ci` 和 web build。这一步需要登录节点能上网。
- **作业里的 venv**：每个作业都在 `$SLURM_TMPDIR/venv` 里现建。修复作业装 `requirements.txt` 和 `scripts/exp900/requirements-900.txt`，识别作业只装后者。不用手动维护。

先核对模块和 wheel。模块只在 `scripts/setup_env.sh` 里加载，版本不对只改这一个文件：

```bash
module spider python/3.12
module spider nodejs                # setup_env.sh 里是 nodejs/24.15.0，Vulcan 没有就改成 spider 列出的版本
avail_wheels torch==2.7.1 torchvision==0.22.1 transformers==4.57.6 peft==0.18.0 accelerate==1.11.0 --python 3.12
avail_wheels playwright==1.57.0 trimesh==4.7.1 scipy==1.18.1 --python 3.12
bash scripts/bootstrap_login.sh
source scripts/setup_env.sh && python -m pytest -q tests/unit      # 应为 113 passed, 1 skipped
```

## 4. 账户与提交参数

Vulcan 要用 AIP RAP 账户，名字以 `aip-` 开头，不是 def-zhouyang。作业脚本里不写账户和 GPU，全部在命令行给。

```bash
sacctmgr -n show assoc user=$USER format=account%30 | sort -u      # 找 aip-xxxx
sinfo -o '%P %G' | sort -u | head                                   # 确认 GRES 名，通常是 gpu:l40s:4
export ACCT=aip-xxxx                                                # 换成上面查到的
export TIO_RUNS=$PWD/tio-runs TIO_MODELS=$PWD/models
CPU_JOB="--account=$ACCT --output=$TIO_RUNS/logs/%x-%j.out --cpus-per-task=4 --mem=16G"
GPU_JOB="--account=$ACCT --output=$TIO_RUNS/logs/%x-%j.out --gpus-per-node=l40s:1 --cpus-per-task=8 --mem=64G"
PACK=$TIO_RUNS/data900/repair58/three-tier-900-validation-repair-58-v1
EXP=$TIO_RUNS/data900/three-tier-900-sft-v1/episode_export
```

以下所有 `sbatch` 都在仓库根目录执行，作业用 `$SLURM_SUBMIT_DIR` 定位仓库。

## 5. 生成器网络与冒烟（硬检查点）

Vulcan 计算节点不能直接上网，只能走默认的 Squid 代理，白名单之外的域名不通。

**第一步，先试白名单。** 中转站主机就是 `.env` 里 `ANTHROPIC_BASE_URL` 的主机部分，下面的命令只取主机名，不打印 key。

```bash
HOST=$(grep '^ANTHROPIC_BASE_URL=' .env | sed -E 's#^[^=]*=https?://([^/:]+).*#\1#')
sbatch $CPU_JOB --time=00:10:00 --job-name=probe --wrap "curl -sS -o /dev/null -m 15 -w 'HTTP %{http_code}\n' https://$HOST/"
```

日志里有 HTTP 码（哪怕是 403 或 404），就说明白名单已经通，后面不用设 `GEN_PROXY`。超时或者 `000` 就是不通，转第二步。

**第二步，不通就在登录节点起转发。** 这是 Vulcan 上的常用做法。脚本只用标准库，只放行这一个主机的 443 端口，必须带令牌，TLS 端到端，登录节点只转发字节，负载很轻。

```bash
tmux new -s tioproxy                        # 记下当前登录节点主机名，转发只在这台上
cd ~/scratch/TurnItOver && source scripts/setup_env.sh
HOST=$(grep '^ANTHROPIC_BASE_URL=' .env | sed -E 's#^[^=]*=https?://([^/:]+).*#\1#')
python scripts/exp900/connect_proxy.py --allow-host "$HOST" --port 0 --token-file ~/.tio-proxy-token
# 打印一行 GEN_PROXY=http://tio:<token>@<登录节点>:<端口>，Ctrl-b d 退出 tmux，转发继续运行
export GEN_PROXY='http://tio:<token>@<登录节点>:<端口>'     # 复制上面那一行
```

注意几点：
- 打印出来的主机名必须能从计算节点解析。不能的话，换成 `hostname -i` 得到的内网地址。
- 同时给 Vulcan 技术支持发邮件，申请把这个域名加入白名单，这是正规途径。
- 实验结束后在 tmux 里 Ctrl-c 停掉转发。

**第三步，生成器冒烟，2 例，judge 用 gold 和 none。** 不需要 GPU。

```bash
CASES=$PACK OUT=$TIO_RUNS/repair/smoke-gen JUDGES="gold none" GENERATOR=api LIMIT=2 GEN_PROXY=${GEN_PROXY:-} \
  PROBE_URL=https://$HOST/ sbatch $CPU_JOB --time=01:00:00 --job-name=smoke-gen scripts/slurm/repair_table.sbatch
```

通过的标准：
- 日志里每个 episode 都打出一行 JSON，`model_calls` 不小于 1。
- `$TIO_RUNS/repair/smoke-gen/k3/gold/*/round-000/model.json` 里 `finish_reason` 正常，usage 有 token 数。
- 记下每轮耗时和 token，参考值大约是每轮 4 万 token，20 到 60 秒。

如果日志里是 `HTTP 429` 或 `HTTP 503`，说明中转站本身有问题，和转发无关，按最后一节换生成器。

## 6. 校准与资格集（CPU，不需要 API）

scripted-stop 不改程序，scripted-fix 直接换成参考程序，两者都不是模型。由此定出所有 judge 共用的资格集：
- 未修改的候选程序必须审计不通过，也就是缺陷可见。
- 换成参考程序必须审计通过。

```bash
for g in scripted-stop scripted-fix; do
  CASES=$PACK OUT=$TIO_RUNS/repair/calib-$g JUDGES=none GENERATOR=$g \
    sbatch $CPU_JOB --time=03:00:00 --job-name=calib-$g scripts/slurm/repair_table.sbatch
done
# 两个都跑完后（登录节点，很轻）：
source scripts/setup_env.sh
python scripts/exp900/eligible_cases.py --stop $TIO_RUNS/repair/calib-scripted-stop/scripted-stop/none \
  --fix $TIO_RUNS/repair/calib-scripted-fix/scripted-fix/none --write $TIO_RUNS/repair/eligible.txt
```

Nibi 登录节点上前 16 例的校准结果见本文末尾。如果 58 例中被排除的超过 5 例，先查原因再往下走。

## 7. GPU judge 冒烟（2 例，脚本生成器）

```bash
CASES=$PACK OUT=$TIO_RUNS/repair/smoke-gpu JUDGES="zs8b_sweep ours_2b_evidence" GENERATOR=scripted-stop LIMIT=2 \
  sbatch $GPU_JOB --time=01:00:00 --job-name=smoke-gpu scripts/slurm/repair_table.sbatch
```

通过的标准：
- `$TIO_RUNS/repair/smoke-gpu/env-*.txt` 里能看到 L40S、adapter 的路径和 torch 版本。
- 每个 `round-000/verdict.json` 都存在。
- `judge_call.json` 里 `validation_error` 为空，或者是能解释的 uncertain。

## 8. 5 例 pilot（6 个 judge 全跑）

从资格集里取 2 个 easy、2 个 medium、1 个 hard：

```bash
python3 - <<EOF > $TIO_RUNS/repair/pilot.txt
import json
ok = [l.strip() for l in open("$TIO_RUNS/repair/eligible.txt") if l.strip() and not l.startswith("#")]
diff = {c["case_id"]: c["difficulty"] for c in json.load(open("$PACK/cases.json"))}
pick = [c for c in ok if diff[c] == "easy"][:2] + [c for c in ok if diff[c] == "medium"][:2] + [c for c in ok if diff[c] == "hard"][:1]
print("\n".join(pick))
EOF
for j in none pixel gold; do
  CASES=$PACK OUT=$TIO_RUNS/repair/pilot/$j JUDGES=$j GENERATOR=api CASE_IDS_FILE=$TIO_RUNS/repair/pilot.txt GEN_PROXY=${GEN_PROXY:-} \
    sbatch $CPU_JOB --time=02:00:00 --job-name=pilot-$j scripts/slurm/repair_table.sbatch
done
for j in zs8b_sweep ours_2b_ordinary ours_2b_evidence; do
  CASES=$PACK OUT=$TIO_RUNS/repair/pilot/$j JUDGES=$j GENERATOR=api CASE_IDS_FILE=$TIO_RUNS/repair/pilot.txt GEN_PROXY=${GEN_PROXY:-} \
    sbatch $GPU_JOB --time=02:00:00 --job-name=pilot-$j scripts/slurm/repair_table.sbatch
done
# 全部跑完后：
python scripts/summarize_repair_table.py --out $TIO_RUNS/repair/pilot/* --cases $PACK --labels $EXP/annotations.private.jsonl \
  --case-ids-file $TIO_RUNS/repair/pilot.txt --summary $TIO_RUNS/repair/pilot-summary.json
```

pilot 通过的标准：
- gold 的 Success 至少过半，并且不低于其他所有行。
- none 不高于 gold。
- 没有 episode 因为异常没跑完，`result.json` 的 status 都是 complete。

如果 gold 和 none 差不多，说明协议或生成器有问题，不要开整表，先看 `round-*/prompt.txt` 和 `response.txt`。

## 9. 冻结后跑整表

最晚 11-16 06:00 EST 冻结，之后不再改代码。每个作业会在 `env-<jobid>.txt` 里记下 commit 和未提交改动数，冻结时应为 `dirty=0`。

```bash
for j in none pixel gold; do
  CASES=$PACK OUT=$TIO_RUNS/repair/full/$j JUDGES=$j GENERATOR=api CASE_IDS_FILE=$TIO_RUNS/repair/eligible.txt GEN_PROXY=${GEN_PROXY:-} \
    sbatch $CPU_JOB --time=08:00:00 --job-name=full-$j scripts/slurm/repair_table.sbatch
done
for j in zs8b_sweep ours_2b_ordinary ours_2b_evidence; do
  CASES=$PACK OUT=$TIO_RUNS/repair/full/$j JUDGES=$j GENERATOR=api CASE_IDS_FILE=$TIO_RUNS/repair/eligible.txt GEN_PROXY=${GEN_PROXY:-} \
    sbatch $GPU_JOB --time=08:00:00 --job-name=full-$j scripts/slurm/repair_table.sbatch
done
```

运行特点：
- 每个作业是 58 例乘最多 3 轮，生成器最多约 170 次调用，预计 1 到 3 小时。6 个作业合计约 35 到 45M token。
- runner 可以续跑：已完成的 episode 会跳过，没跑完的会改名保留。作业超时后用同样的命令再交一次即可。
- 生成器的回复按提示词和图片哈希缓存在各自 `OUT/cache/generator/`，重交不会重复花 token。
- 如果 11-16 08:00 EST 还没开跑，就从 `eligible.txt` 按层固定取 30 例，写成 `eligible30.txt`，所有 judge 都换用它。
- 时间够再加 `zs8b_screenshot` 和 `ours_8b_evidence`，写法和上面的 GPU 循环一样。

## 10. 识别补全（GPU，和整表并行）

L40S 比 H100 慢 2 到 3 倍，所以时间给宽一些。

```bash
# 8B Ordinary 第三个 seed：训练加验证集评测，Nibi H100 上约 25 分钟
MODE=sft SIZE=8B SEEDS=20261213 SUPS=ordinary sbatch $GPU_JOB --time=03:00:00 --job-name=sft8b-ord scripts/slurm/exp900.sbatch
# withheld：基座加该尺寸全部 adapter，在去掉暴露格的 47 例上评测
for s in 2B 8B; do SIZE=$s sbatch $GPU_JOB --time=04:00:00 --job-name=withheld-$s scripts/slurm/exp900_withheld.sbatch; done
# 可选：测试集 180 例，只评已有 checkpoint，不重训
for s in 2B 8B; do MODE=eval SIZE=$s SPLIT=test sbatch $GPU_JOB --time=04:00:00 --job-name=test-$s scripts/slurm/exp900.sbatch; done
```

通过的标准是各 run 目录的 `status.tsv` 每行都是 ok。

## 11. 汇总与交回

```bash
source scripts/setup_env.sh
python scripts/summarize_repair_table.py --out $TIO_RUNS/repair/full/* --cases $PACK --labels $EXP/annotations.private.jsonl \
  --case-ids-file $TIO_RUNS/repair/eligible.txt --summary $TIO_RUNS/repair/full-summary.json > $TIO_RUNS/out/RESULTS-repair.md
python scripts/exp900/summarize_900.py $TIO_RUNS/out > $TIO_RUNS/out/RESULTS-level900-table.md
python scripts/exp900/summarize_900.py $TIO_RUNS/out test > $TIO_RUNS/out/RESULTS-level900-test.md       # 跑了测试集才有
python scripts/exp900/analysis/withheld_summary.py   # 自己写入 $TIO_RUNS/out/RESULTS-withheld.md
python scripts/exp900/analysis/unexposed_rate.py     # 自己写入 $TIO_RUNS/out/RESULTS-grounding-offline.md
```

交回方式：
- 把 `RESULTS-*.md`、`eligible.txt` 和 `full-summary.json` 复制到仓库的 `docs/results/`，在 `exp/vulcan-900` 分支上提交并推送。这些文件不含凭证。
- 定性图需要 2 到 3 个完整 episode 目录，例如 `repair/full/ours_2b_evidence/k3/ours_2b_evidence/<case>/`，用 Globus 传回 Nibi 或个人机器。

## 时间线（EST）

| 时间 | 事情 |
|---|---|
| 11-15 晚 | 搬运，环境，网络，生成器冒烟，校准，交识别作业 |
| 11-16 06:00 前 | GPU 冒烟，5 例 pilot，冻结 |
| 11-16 06:00 到 12:00 | 修复整表 |
| 11-16 12:00 以后 | 汇总交回，写论文 |

## Plan B

- **中转站持续 429 或 503**：
  - 改用 `.env` 里 OpenAI key 对应的模型，设 `TIO_GENERATOR_PROVIDER=openai` 和 `TIO_GENERATOR_MODEL=<模型名>`，模型名先核实再写进论文。
  - 转发要另起一个实例，放行 `OPENAI_BASE_URL` 的主机。
  - 换生成器后，第 5 到 8 步要重做。
- **转发和白名单都不通**：在 Vulcan 本地用 vLLM 起开源 VLM，作为 OpenAI 兼容接口。`turnitover/models/client.py` 支持 `openai_compatible`。权重要提前在登录节点下载。
- **Nibi 提前恢复**：同一套命令在 Nibi 也能跑，账户换成 `def-zhouyang`，GPU 写法换成 `--gpus-per-node=h100:1`。Nibi 计算节点能直接上网，不需要 `GEN_PROXY`。

## Nibi 登录节点上的校准（前 16 例）

11-15 在 Nibi 登录节点用 `configs/repair_pack900.yaml` 对按 case id 排序的前 16 例做了校准（easy 14, hard 2），judge 为 `none`，不涉及模型和 API。

- scripted-stop：16 例中审计通过 0 例，即未改动的候选程序全部被独立审计判为不通过，缺陷对审计可见。每例平均 7.3 秒。
- scripted-fix：16 例中审计通过 16 例，即换成参考程序后全部通过。每例平均 6.4 秒。
- 资格集 16/16。早先冒烟里 hp-02457e8ca38131 恢复参考后只到 0.944，是因为那次误用了 ReplicaCAD 配置（三角形上限 5000）；用 900 修复包配置（上限 2,000,000）后通过，所以成功判定不用改。
- 剩下 42 例按第 6 步在 Vulcan 上补齐。
