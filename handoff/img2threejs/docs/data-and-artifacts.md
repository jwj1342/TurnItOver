# Data and artifacts

本仓库同时处理 Viewer 会话数据、实验输入资产和本地运行产物。三者的版本化策略与可信语义不同，不应混用。

## 数据目录

```text
data/
├── sessions/       # Chatbox Viewer 输入，可被版本化
├── assets/         # 受控实验输入资产，可被版本化
├── runs/           # 本地运行产物，被 gitignore
└── bundles/        # 可选的 Chatbox 分片数据包
```

旧 `data/raw_sessions/` Codex JSONL 流程不再属于当前数据接口；Viewer 不会发现或解析该目录。

### `data/sessions/`

每个目录必须包含 Chatbox `session.json`，图片位于 `resources/`：

```text
data/sessions/<session-id>/
├── session.json
└── resources/
    ├── input.png
    ├── resource-000001.png
    └── ...
```

支持 `.png`、`.jpg`、`.jpeg`、`.webp`。参考图优先识别 `input.*`，并兼容历史拼写 `inpuy.jpg`。`resource-*` 按末尾数字排序；与 reference 哈希相同的副本会被排除。

Parser 只读取顶层 `messages`，不解析 `messageForksHash`。不要为了 Viewer 修改原始 `session.json`。

### `data/assets/`

用于保存可复用、来源明确的实验输入。当前仓库包含：

```text
data/assets/scissors_demo/reference.jpg
data/assets/scissors_demo/screenshots/
```

`reference.jpg` 可作为文档命令中的真实示例输入。screenshots 不应自动解释为某次当前 Pipeline run 的结果，除非结果文档明确给出 provenance。

### `data/runs/`

这是 runtime、Codegen、Verifier 和完整 Pipeline 的默认输出位置，当前被 `.gitignore` 排除。目录存在不表示结果已经审计、可复现或支持研究结论。

## Runtime smoke 输出

```text
data/runs/runtime-smoke/
├── render.png
├── 01_orbit_right.png
├── 02_zoom_in.png
└── result.json
```

若 render check 失败，可能只有 `render.png`（仅在找到 canvas 时）和 `result.json`。

## Codegen 输出

参考图扩展名会被保留：

```text
data/runs/<run-id>/
├── reference.<ext>
├── workspaces/
│   ├── attempt_00/
│   └── ...
├── call_00_generate/
│   ├── code.html
│   ├── reasoning.txt
│   └── agent.json
├── call_01_repair/                 # 仅发生 runtime repair 时出现
├── render_attempts/
│   ├── attempt_00/
│   │   ├── candidate.html
│   │   ├── render.png              # 找到 canvas 时出现
│   │   └── check.json
│   └── ...
├── final.html
├── trajectory.json
└── summary.json
```

`reasoning.txt` 只保存模型/OpenHands 实际暴露的 reasoning，不尝试恢复隐藏推理。

## 独立 Verifier 输出

```text
<out>/
├── verifier/
│   ├── view_00_initial.png
│   ├── view_01_<action-or-pose>.png
│   ├── ...
│   └── trace.json
└── verifier_agent.json
```

`trace.json` 保存 observation、action payload、runtime metadata、feedback、verdict、selected view IDs 和 budget 状态。

若 Verifier 在一次受限收尾重试后仍未调用 `finish`，不会生成伪造 verdict；已有诊断写入 `verifier/incomplete_agent.json`，随后入口明确失败。

## 完整 Pipeline 输出

```text
data/runs/<run-id>/
├── reference.<ext>
├── round_00/
│   ├── workspace_generate/
│   │   ├── candidate.html
│   │   └── reference.<ext>
│   ├── code_agent_generate/
│   │   ├── code.html
│   │   ├── reasoning.txt
│   │   └── agent.json
│   ├── render_attempts/
│   │   └── attempt_00/
│   │       ├── candidate.html
│   │       ├── render.png
│   │       └── check.json
│   ├── runtime_repairs/            # 仅 render fail 后发生修复时出现
│   ├── verifier/
│   │   ├── view_00_initial.png
│   │   ├── view_01_*.png
│   │   └── trace.json
│   └── verifier_agent.json
├── round_01/
│   ├── workspace_visual_revise/
│   ├── code_agent_visual_revise/
│   ├── render_attempts/
│   ├── verifier/
│   └── verifier_agent.json
├── final.html
├── trajectory.json
└── summary.json
```

失败可能使部分可选文件或后续目录不存在。`final.html` 是最后一个 candidate，不代表已被接受。

## 结果判定

完整 Pipeline 的权威摘要是 `summary.json`：

- `accepted`：Verifier 返回 `accept`。
- `max_visual_revisions`：修改预算耗尽，最后一轮仍为 `revise`。
- `render_failed`：某轮 candidate 在 render repair 预算内仍未通过。

`trajectory.json` 提供逐轮证据。正式结论必须同时保留 run 配置、代码 commit、输入资产和结构化状态。

## 隐私与版本化

- `data/sessions/` 当前允许进入 Git，可能包含 prompt、visible reasoning、模型元数据和图片；提交前必须脱敏。
- `.env` 不得提交。
- `data/runs/` 默认不提交，因此本地产物不能自动成为仓库可验证的正式结果。
- 如需发布正式结果，应创建独立结果记录，并明确哪些最小证据被版本化或存储在外部可访问位置。
