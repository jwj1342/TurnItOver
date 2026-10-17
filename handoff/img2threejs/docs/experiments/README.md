# Experiment documentation policy

此目录记录实验 provenance 和结构化结论，不把“目录存在”或“页面能够渲染”自动解释为研究效果成立。

## 正式结果记录模板

每个正式结果至少应包含：

```text
title
date and timezone
git commit and worktree state
input asset path and identity
model/provider identifier (不记录密钥)
relevant non-secret configuration
action space and config
render/visual revision budgets
run ID and artifact location
summary.status
summary.accepted
round count and revision count
known failures or uncertainty
claim supported by this run
claims not supported by this run
```

若产物位于被 gitignore 的 `data/runs/`，记录必须明确说明其他读者是否能够取得原始证据。

## `pipeline-demo-001` 说明性示例

当前本地工作区存在被 gitignore 的 `data/runs/pipeline-demo-001/`。它只用于解释 Pipeline schema 和停止状态，不是正式 benchmark、成功 demo 或效果证据。

其可读摘要为：

```json
{
  "status": "max_visual_revisions",
  "accepted": false,
  "visual_revisions": 2,
  "rounds": 3
}
```

三轮均产生了 render-check 和 verifier 产物，但最终 Verifier 仍要求修改。正确解读是“在给定预算内未被接受”，不能写成“Pipeline 成功重建”或“Verifier 验证通过”。

该目录未被版本化，也没有在本文档中声明完整模型配置和可复现环境，因此不得从它推导质量提升、成功率或物理合理性结论。

## 最低证据规则

- `render_check.success=true` 只证明页面通过当前确定性可渲染门槛，不证明视觉或物理正确。
- 代码中存在某组件不等于该组件在选定视角中清晰可见。
- `accepted=true` 是当前 Verifier 在给定 observation budget 下的工程停止信号，不是经过校准的通用质量标签。
- 单个样例不能支持跨数据集、跨模型或跨 action policy 的比较结论。
