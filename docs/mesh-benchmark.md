# 真实网格 benchmark 的构建与验收

2026-09-12，已将 ReplicaCAD 六个资产族（柜子、抽屉柜、门、冰箱、厨房台柜、厨房吊柜）
接入既有腐蚀、浏览器、确定性检查和数据导出链路。当前关注结构与运动学，不能用于材质或物理保真结论。

## 转换与独立审计

```bash
source scripts/setup_local.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/convert_replicacad.py
python scripts/audit_replicacad.py --out output/replicacad-fk-v3
```

输入默认为已获取的 `data/assets/replicacad-3e8c7fe/`；转换结果位于
`data/assets/replicacad-catalog-v1/`。转换器核对 acquisition 文件哈希，输出每个对象的 spec、
catalog 和 conversion 清单。不支持的关节、mimic、远端 mesh URI、视觉几何类型会明确报错。

转换保留 URDF 的 link/joint 树、固定变换、轴、正负限位、多个 visual、GLB 节点变换与正数 scale。
支持 fixed/revolute/prismatic；三角网格内嵌到程序中，执行阶段不联网。
URDF joint origin 属于父坐标系，轴属于关节坐标系，因此必须保留固定旋转再施加关节运动；
定义依据 [ROS 官方 URDF 文档](https://docs.ros.org/kinetic/api/urdf/html/index.html)。

纹理通过 UV 采样成顶点色，并转成线性 RGB；细纹理、高光参数及透明度不保证保留。
没有把视觉网格替换成 box，也没有对网格做简化。碰撞网格、质量、惯量保留在上游文件中但未进入检查器。

审计器直接从原始 URDF 做 FK，再与 Three.js 导出的逐部件顶点和三角形索引比对。
覆盖 rest、每个关节的下限/中点/上限，以及三个固定种子的组合状态。
六资产共 102 状态通过，最大逐顶点误差为 `3.0992777340252657e-7 m`，阈值为 `1e-5 m`。
审计报告记录对应程序 hash 和源文件 hash；后续 benchmark 拒绝缺失或不匹配的程序审计。
`rest.png` 与 `motion.png` 保存真实浏览器图像。
补充审计 `output/replicacad-fk-v3/audit.json` 直接对照上游网格的三角形连接关系，仍全部通过。

## 金标准修正

真实资产试跑发现旧 `point_surface_distance` 的固定 16 个最近三角形中心不是可靠的表面距离近似：
大三角形的中心可能很远，导致同一网格对自身的 Chamfer 也超过 5 mm；退化面还会导致 NaN。

修正为保守包围球筛选候选三角形，再计算点到三角形的距离。筛选上界来自最近的被引用顶点，
不会因固定候选数量漏掉近邻大面；退化三角形按线段/点计算。与暴力遍历和解析案例的测试已通过。
这修正了检查器，而不是降低阈值或删掉失败资产。旧几何结果需重新计算，不能与修正后的数值直接混用。

## 数据试跑与完整流水线

```bash
python -m turnitover generate --config configs/generate_replicacad.yaml \
  --n-samples 12 --out data/runs/replicacad-pilot-v2
```

修正后六资产的 12 样本试跑全部执行成功并通过标签资格检查。原失败试跑
`data/runs/replicacad-pilot-v1/` 保留了失败清单，不能用于训练或评测。

完整小型 benchmark 配置为 96 样本，正常概率 25%，其余 1–2 种腐蚀：部件偏移与关节轴错误。
现有三角形预算腐蚀只支持 box 细分，不能静默用于 mesh，所以此配置不包含该类别。
观测覆盖所有关节的上下限和中点；`zero` 仍表示下限，不等于 rest，`limit` 也不必然等于打开状态。

```bash
sbatch --account=def-zhouyang --job-name=tio-mesh-benchmark --time=01:00:00 \
  --cpus-per-task=2 --mem=8G --export=ALL,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 \
  scripts/slurm/run_portable.sbatch .venv-local/bin/python scripts/run_mesh_benchmark.py
```

脚本依次检查转换审计、生成样本、按资产组固定划分、隔离导出、运行保存观测上的 runtime 基线。
输出根目录为 `data/benchmarks/replicacad-v1/`，最后的 `benchmark.json` 只有所有步骤成功才标记 complete。
任一步失败保留 incomplete 状态，不覆盖旧产物。提交记录：Slurm job `21782957`；应查询调度器和
`benchmark.json` 确认最终状态，不能从提交成功推断实验完成。

本次任务已由 `sacct` 确认为 COMPLETED / ExitCode 0:0，用时 5 分 4 秒。
独立读取原始与导出清单确认：96 个合格样本，train/validation/test 分别为 64/16/16，未隔离样本。
任务使用 v2 审计；之后同一批转换程序也通过上述 v3 拓扑审计。
runtime 测试集的 16 个判定全部为 uncertain，其中关节轴标签 12 个漏检、部件偏移标签 11 个漏检。
这符合该基线只检查预算的能力范围，不能解释为视觉 verifier 的测试结果。

可核验产物：`data/benchmarks/replicacad-v1/benchmark.json`、`raw/shard-0000.manifest.json`、
`splits.json`、`prepared/*.inputs.jsonl`、`prepared/*.gold.jsonl` 和 `runtime-test/metrics.json`。
测试覆盖：87 个单元测试与 11 个浏览器测试（全量 10 个通过后，新增上游拓扑回归及原 FK 测试两项通过）。

## 后续进展与剩余目标

本页记录 benchmark 构建阶段，随后已完成[动作教师与训练导出](oracle-supervision.md)、
[Qwen 分层能力诊断](capability-diagnosis.md)和[真实模型修复 pilot](repair-experiment.md)。
这些结果的轻量测量已纳入 [Git 快照](../results/2026-09-12/README.md)。

仍缺少单参考最终诊断/停止监督、实际训练、多关节留出族、逐轮重算的完整 RP oracle 上界
和真实照片评估。六个资产族只构成小型基准，不证明广泛类别泛化。
完整下一步顺序见 [研究状态](research-status.md)。
