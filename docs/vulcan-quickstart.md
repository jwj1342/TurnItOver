# Vulcan 快速指南（可直接粘贴给 Vulcan 上的 Claude Code）

你在 Vulcan 集群上接手 TurnItOver 的截稿实验，截止时间是 11-17 06:59 EST。详细步骤和每一步的检查点在 `docs/vulcan-handoff.md`，本页是精简版，冲突时以那份为准。

## 规则

- **登录节点**只做 git、编辑、查询、小于 16 例的冒烟和转发服务。训练、评测、修复一律用 `sbatch`，并且在仓库根目录提交。
- **模块**只在 `scripts/setup_env.sh` 里加载。Python 包只从 Alliance wheelhouse 装（`pip install --no-index`），不用 conda、uv 或 PyPI。
- **查询 Slurm** 时，两次之间至少隔 60 秒，不要写轮询循环，也不要在作业里插 sleep。
- **不改实验代码**。确实需要修，先停下来说明原因。凭证在 `.env` 里，不打印、不提交。
- **结果**写到 `tio-runs/` 下面。汇总好的 `RESULTS-*.md` 复制到 `docs/results/`，提交到分支 `exp/vulcan-900` 并推送。

## 0. 同步代码

文件夹是从 Nibi 用 Globus 整体传过来的，代码以 git 为准：

```bash
cd <Vulcan 上的 TurnItOver 路径>
git fetch origin && git checkout exp/vulcan-900 && git reset --hard origin/exp/vulcan-900   # 这里还没有本地工作，只是同步
rm -rf .venv && chmod 600 .env                                                             # Nibi 的 venv 在这里不能用
```

## 1. 登录节点体检

```bash
bash scripts/exp900/vulcan_preflight.sh
```

它检查这些内容，并把报告写到 `tio-runs/logs/preflight-*.txt`：
- git 状态。
- 传输完整性：对照 `scripts/exp900/transfer-manifest.tsv` 逐文件核对大小，小文件核对 SHA256。必须看到 `TRANSFER OK`。
- `.env`、模块、wheel。
- 账户、分区、GPU 类型、每用户的作业和资源上限、fair share。
- 登录节点能否访问生成器主机。

报告里每条 `ACTION` 和 `MISSING` 都要先处理。最后几行给出建议的 `ACCT`、`CPU_JOB`、`GPU_JOB`，照着 export。Vulcan 的账户以 `aip-` 开头，GPU 是 L40S 48GB，每节点 4 张，最长 7 天。

## 2. 环境

```bash
bash scripts/bootstrap_login.sh
source scripts/setup_env.sh && python -m pytest -q tests/unit
```

bootstrap 会建登录节点的 `.venv`，下载 Playwright 浏览器，并执行 web build。单测应为 116 passed、1 skipped。作业不复用这个 venv，每个作业都在 `$SLURM_TMPDIR` 里从 wheelhouse 现建。模块缺失时只改 `scripts/setup_env.sh`。

## 3. 网络架构与计算节点探针

Vulcan 的计算节点不能直接上网，只有一个集群默认的 Squid 代理，按白名单放行；登录节点可以上网。生成器 API（`.env` 里 `TIO_GENERATOR_PROVIDER` 对应的 BASE_URL 主机）有三种接法：

1. **白名单已经放行**：什么都不用做。
2. **登录节点转发**：在 tmux 里运行
   ```bash
   python scripts/exp900/connect_proxy.py --allow-host <该主机> --port 0 --token-file ~/.tio-proxy-token
   ```
   - 它只放行这一个主机的 443 端口，必须带令牌，TLS 端到端。
   - 把打印出来的 `GEN_PROXY=...` export 出来，修复作业会自动带上。
   - 主机名如果计算节点解析不了，就换成 `hostname -i` 的地址。
   - 同时给 Vulcan 支持发邮件，申请把该域名加入白名单。
3. **都不通**：按 handoff 文档的 Plan B，在本地用 vLLM 起开源模型。

用一个 20 分钟的 CPU 作业测清楚走哪条路，顺便测浏览器渲染：

```bash
sbatch $CPU_JOB --time=00:20:00 --job-name=probe scripts/slurm/vulcan_probe.sbatch            # 有 GEN_PROXY 时会一并测转发
```

日志里 `HTTP 000` 表示不通，其他任何状态码都表示通了。`browser ok` 必须出现。

## 4. 并发怎么定

- **GPU**：每个 GPU judge 作业只要 1 张 L40S。可以同时跑几个，看体检报告里的 `MaxTRESPU` 和 `MaxJobsPU`，以及 `squeue` 里的排队情况。
- **生成器**：并发数就是同时在跑的 API 作业数，runner 在每个作业里是串行的。先同时放 3 个，日志里没有 `generator_retry` 再放满 6 个。
  - runner 对 429 和 5xx 会退避重试 4 次，大约 5 分钟。
  - 重试仍失败作业就会退出。原样重交即可续跑，已完成的 episode 和生成器回复都有缓存，不会重复花 token。
- **识别作业**（8B 补 seed、withheld）和修复作业互不依赖，可以同时排队。

## 5. 按顺序跑实验

命令全部在 `docs/vulcan-handoff.md` 的第 5 到 11 步。顺序和检查点如下：

| 步骤 | 内容 | 检查点 |
|---|---|---|
| 5 | 生成器冒烟，2 例，judge 用 gold 和 none | `model.json` 正常。如果是 429 或 503，说明中转站坏了，转 Plan B，也就是换 `.env` 里 OpenAI key 的模型，模型名先找人确认 |
| 6 | 58 例 scripted-stop 和 scripted-fix 校准，CPU | 生成 `eligible.txt`，Nibi 上前 16 例全部合格 |
| 7 | GPU judge 冒烟，2 例 | `env-*.txt` 里有 L40S，`verdict.json` 存在 |
| 8 | 5 例 pilot，6 个 judge | gold 的成功率不低于其他行，并且过半 |
| 9 | 11-16 06:00 EST 前冻结，然后跑整表 | 6 个 judge 各一个作业，只用资格集。11-16 08:00 还没开跑就改用固定的 30 例子集 |
| 10 | 识别补全 | 8B Ordinary seed 20261213、withheld 2B 和 8B，可选测试集评测。`status.tsv` 每行都是 ok |
| 11 | 汇总 | `RESULTS-repair.md` 等结果提交到 `docs/results/` 并推送，再挑 2 到 3 个完整 episode 目录留作定性图 |

遇到检查点不过，先停下来报告现象、日志路径和你的判断，不要自行改实验设计。
