# 三状态成对观察 SFT

本分支基于 `laq/three-tier-900-sft-export`（`88ac0df`）。新 release exporter 默认将 Easy、Medium、Hard 统一为 **一个视角 × 三个状态 × reference/candidate = 六张图、三对观察**。不再单列共享参考图，不把参考帧混作 candidate history。

| observation / evidence_steps | 状态 | 图片顺序 |
|---|---|---|
| 0 | zero（旧文件名 rest/reset） | reference_zero, candidate_zero |
| 1 | mid | reference_mid, candidate_mid |
| 2 | limit | reference_limit, candidate_limit |

`observation_format=reference_candidate_pairs_v1` 明确区分旧格式。每个 history 项包含 step、state、action.view_id、reference_image_ref 和 image_ref（candidate）。`image_order` 必须等于按每对 reference/candidate 展开的六图清单。模型输入为每对显式标注 observation index、state、view、reference/candidate 的文本与图像序列；本地训练、评测和 PyTRIO 编码共用该序列。引用 1 代表整组 mid 对比，不能解释为第1张图。

## 两种监督及参数

Ordinary/Evidence 使用相同公开输入、图片字节和顺序，各自从同一 base model 创建独立 LoRA adapter。Ordinary 使用完整注入标注，引用全部三对；Evidence 只监督受支持标签，并精确引用支持该标签的 pair indices。私有 `evidence.pair_label_indices` 用于校验，不能进入 prompt。

新构造要求单视角暴露所有注入缺陷，因此合格样本的两种监督缺陷集合相同，主要差别在证据引用及回答文本；不能沿用旧实验中“Ordinary 额外监督不可见缺陷”的解释。像素差分仍是代理曝光，不是人工可识别性认证。无注入标签仍不构成正向 pass 证据，clean 目标保持 uncertain。

本地训练配置保持 lr=2e-4、epochs=3、LoRA rank=8 / alpha=16、dropout=0、q/k/v/o projection、AdamW、assistant-only token-mean CE、每样本一次更新、梯度裁剪1.0。基座权重冻结，保存adapter。评测保持greedy、最多512 tokens、按来源物体bootstrap 1000。PyTRIO 编码支持同一六图输入与assistant-only causal CE；服务端公开训练接口未暴露alpha，本分支不宣称其alpha等于16，不启动任何服务训练。PyTRIO模型为Qwen3.5-4B；本地原入口的Qwen3VL模型类不等于Qwen3.5，不应直接替换模型路径声称兼容。

## 数据构造与审计

Easy旧release只有rest/limit四图。通过 `--easy-board-root`（可重复）提供原始912×630原生展示板；四张端点tile必须与release图片逐像素一致，才裁取两张256×256 mid图。记录board SHA、裁剪框和端点SHA。拒绝插值、缩放或用端点复制mid。

Medium使用原生三状态六帧。Hard对已有两个视角分别审核，选择能覆盖全部注入标签且总像素差分最高的一个视角；clean按同样确定性规则选择。此选择是固定数据构造，使用私有标注，不是模型主动选视角能力或新测试结果。

单缺陷可用同状态reference/candidate像素差分作为代理。复合缺陷不能把整体差分归给所有标签，必须用 `--exposure-audit` 提供逐标签审核/反事实证据。结构为case ID → views → view ID → 记录：

```json
{
  "case-id": {"views": {"front": {
    "reference_sha": "16 hex", "candidate_sha": "16 hex",
    "image_sha256": ["R_zero SHA", "C_zero SHA", "R_mid SHA", "C_mid SHA", "R_limit SHA", "C_limit SHA"],
    "labels_sha256": "sha256(json.dumps(normalized_labels, sort_keys=True).encode())",
    "pair_label_indices": [[0], [0, 1], [1]],
    "method": "per-label counterfactual rendering or reviewed exposure",
    "provenance": "path/hash of independent evidence"
  }}}
}
```

审核记录绑定原程序、六图和规范化标签，选中记录及canonical SHA写入 `paired-input-audit.private.jsonl`。哈希只验证输入身份，不代替审核。无合格单视角、缺少真实mid或缺少复合逐标签证据时明确拒绝导出，绝不静默丢弃案例凑数。冻结split、来源身份和900配额保持原约束；失败目录不能用于训练，完整manifest仅在全部成功后写出。

```bash
python scripts/build_three_tier_900_sft_export.py \
  --release /private/easy-900 --release /private/medium-900 \
  --hard-candidate-zip /private/hard-v11/private-candidates.zip \
  --split-plan /private/split-plan.json --asset-root /private/exact-source-specs \
  --easy-board-root /private/easy-boards \
  --exposure-audit /private/per-label-exposure.json \
  --output output/datasets/three-tier-paired-v1 --dataset-version three-tier-paired-v1
```

本次交付仅为SFT算法及输入转换代码，不构造、修复或发布全量数据集，不启动训练。输入材料不满足六图契约时明确报错，由数据流程另行补齐。私有数据和临时检查产物不推送Git。

## PyTRIO 训练入口

`turnitover.oracle.pytrio_adapter.model_input` 可生成训练Datum或采样ModelInput。训练入口复用该编码，使用三个epoch，每样本一次优化更新：

```bash
python scripts/train_pytrio_paired_sft.py \
  --dataset /private/paired-export --dataset-manifest-sha256 MANIFEST_SHA256 \
  --out output/pytrio-evidence --supervision evidence \
  --model Qwen/Qwen3.5-4B --epochs 3 --learning-rate 2e-4 --rank 8
```

将 `--supervision` 改为 `ordinary`，使用另一个输出目录训练独立adapter。显式传入 `--alpha` 会拒绝执行，避免静默偏离alpha16协议。该命令会发出训练请求，本次未执行。
