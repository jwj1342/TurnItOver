# 原生照片重建迭代闭环

`turnitover iterate` 将照片到三维程序的生成、可执行性门槛、主动验证和有限轮修复接入 TurnItOver 的正式运行时。它复用同一套 Program ABI、`ObservationSession`、固定视角、关节动作、runtime query、模型 client 与 verifier decision schema，不再由 handoff prototype 维护第二套观察执行逻辑。

## 控制流

```text
reference image
  → generator → ObjectProgram (TypeScript Program ABI)
  → render gate: compile → load → fixed view → non-blank pixels
      └─ failed → bounded runtime repair → render gate
  → fresh verify: active/fixed/random/runtime policy
      ├─ pass → accepted
      ├─ fail → bounded visual revision → render gate → fresh verify
      └─ uncertain/error → explicit terminal status
```

runtime repair 与 visual revision 使用独立预算。前者只接收编译、加载、截图或近似空白画面的结构化错误；后者只接收 verifier 的公开 verdict、findings 及 findings 实际引用的 observations。私有 audit 不进入该模型可见闭环。每次源码改变后都会创建新的 render gate 和 verifier 目录，因此不会复用旧页面、相机、关节状态或对话。

## 运行

先按[模型配置](models.md)设置 generator 与 judge。所有命令从仓库根目录执行，输出目录必须不存在或为空。

```bash
python -m turnitover iterate \
  --reference-image reference.jpg \
  --out output/photo-iterative-001 \
  --policy active \
  --budget 8 \
  --max-runtime-repairs 3 \
  --max-visual-revisions 2
```

使用已有 ABI 程序可跳过初始生成，但后续修复仍需要 generator：

```bash
python -m turnitover iterate \
  --reference-image reference.jpg \
  --initial-program candidate.ts \
  --out output/photo-iterative-from-candidate
```

`--policy runtime` 不调用 judge，但它只能发现运行时预算违规；没有违规时返回 `verification_uncertain`，不会把视觉质量判为通过。命令只在 `accepted=true` 时返回退出码 0，其他终态返回 1。自动化仍应读取 `result.json`，而不是只看退出码或 `final.ts`。

## 状态与 artifact

根 `result.json` 是权威清单，`kind=iterative_synthesis`，并记录配置、git 状态、输入哈希、各轮候选、调用计数、token 计数、最终状态与 `accepted`。终态如下：

| status | accepted | 含义 |
| --- | --- | --- |
| `accepted` | true | fresh verifier 返回证据关联的 `pass` |
| `max_visual_revisions` | false | verifier 仍为 `fail`，或视觉修改预算内没有有效新候选 |
| `render_failed` | false | render gate 在 runtime repair 预算内仍失败 |
| `verification_uncertain` | false | verifier 明确无法在现有证据下判断 |
| `verification_error` | false | verifier 或观察基础设施未正常完成 |
| `generation_failed` | false | 初始生成或某次 repair 模型调用/响应失败 |

主要目录如下：

```text
result.json
reference-00.jpg
generation/{prompt.txt,response.txt,model.json,program.ts,result.json}
round-000/
  candidate.ts
  render-gate/attempt-000/{candidate.ts,render.png,result.json}
  runtime-repairs/repair-000/...
  verifier/{candidate.ts,result.json,feedback.json,trajectory.jsonl,observations/,model_calls/,index.html}
  visual-revisions/revision-000/...
final.ts
```

`render gate success` 只表示程序可由共享 harness 加载且固定截图不是近似常量图；`accepted` 只表示当前 verifier 在给定输入和预算下选择停止。二者都不是物理正确性、隐藏结构正确性或数据集泛化的证明。

## Viewer

`handoff/img2threejs/app.py` 现在可直接发现 TurnItOver 根目录 `output/` 下带 `kind=iterative_synthesis` 的运行。它读取上述权威 artifact，并将 TypeScript diff、render gate 截图和 verifier evidence 适配到原有只读界面；不会生成另一份 trajectory schema，也不会把 TypeScript 当作独立 HTML 执行。
