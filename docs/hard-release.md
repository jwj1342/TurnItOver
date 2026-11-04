# Hard 候选发布

`hard-candidate-v0.1.0-rc.1` 包含 75 个独立复杂多关节物体的 75 条单缺陷案例，腐蚀类型为旋转轴 3 度偏转、关节锚点 0.5% 包围盒对角线偏移和活动部件 2 度初始旋转，各 25 条。每条案例有参考／候选、初始／补充视角及 rest／mid／limit 三态，共 900 张 256×256 原生图。

这是候选版本，不是冻结 benchmark 或训练导出。随机 10 条原始人工回件为 6 hard、3 clean、1 pending；所有者认为难以识别符合 Hard 定义并批准发布，但原始回件不改写。严格离线审计无错误，保留 3 条视觉支持状态与 exact checker 失败状态不重合的警告。训练清单为空，`training_ready=false`。

完整 data ZIP 含便携 Gold、checker、来源哈希、构造程序 custody registry、自动审计和原始随机回件，仅供协调者／研究使用。构造程序、mesh、URDF、下载缓存、失败样本及历史媒体不在附件内，也不提供独立几何重放能力。review ZIP 只含固定随机 10 条案例的两视角三态图、审核页与空白回件，不含 Gold、来源身份或历史判断。

```sh
python scripts/validate_hard_release.py \
  --manifest benchmark/releases/hard-candidate-v0.1.0-rc.1.json \
  --assets data/TIO-datasets/releases/hard-candidate-v0.1.0-rc.1 \
  --output data/verified/hard-rc1
```

输出目录必须不存在。验证器检查公开固定契约、ZIP SHA、文件库存、75 个五键来源身份、三类腐蚀平衡、900 张原生图、答案隔离、人工回件统计和训练空清单。便携验收不会重新运行构造程序或 mesh checker。
