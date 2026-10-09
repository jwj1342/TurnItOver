# UniPhys E2E 接入

## 范围

本次接入使用 UniPhys-Bench Part 2 的 URDF 和部件 Mesh 生成参考图像，并将参考图像输入 `turnitover iterate`，运行 Generator → Render Gate → Verifier → Repair 闭环。

数据集为 [`breezexian/UniPhys-Bench-Part2`](https://huggingface.co/datasets/breezexian/UniPhys-Bench-Part2)，固定 revision 为 `ea633b5c2b1cf9fefb24fb8c30c3703cc75ae7c4`，许可证为 CC-BY-NC-4.0。本轮选择 `UPB_00000000` 至 `UPB_00000019`，共 20 个物体。

## 数据与产物

| 内容 | 位置 |
| --- | --- |
| URDF、部件 Mesh、纹理和下载清单 | `data/assets/uniphys/` |
| Three.js 参考图与渲染清单 | `data/references/uniphys/` |
| 20 个闭环任务的结果 | `output/uniphys-qwen37/` |

`data/` 和 `output/` 由 `.gitignore` 排除。仓库提交接入代码、测试和本文档；完整数据与运行产物通过单独的共享入口提供。

## 运行

从仓库根目录执行：

```bash
python scripts/download_uniphys.py \
  --ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --out-dir data/assets/uniphys

python scripts/render_uniphys_references.py \
  --root data/assets/uniphys \
  --out data/references/uniphys \
  --ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --size 640

python scripts/run_uniphys_iterate_batch.py \
  --references data/references/uniphys \
  --out output/uniphys-qwen37 \
  --env-file .env \
  --expected-model qwen3.7-plus \
  --view 06 \
  --budget 8 \
  --seed 0 \
  --size 384 \
  --max-runtime-repairs 3 \
  --max-visual-revisions 2
```

参考渲染为每个物体生成 12 个静态视角，并为具有有限运动范围的关节生成上限状态图。本轮共记录 240 个静态视角和 22 个关节状态视角。批处理从每个物体选择 `screenshots/06.png`，因此闭环输入为 20 张 PNG，对应 20 个独立物体。

UniPhys 中的 continuous joint 没有有限上限。静态参考渲染将这类关节固定在 URDF 零位；revolute 和 prismatic joint 保留其有限运动范围。

## 当前结果

2026-10-09 的批量运行完成 20 个任务，其中 4 个最终被 Verifier 接受。运行使用 `qwen3.7-plus` 作为 Generator 和 Judge，配置为 active policy、观察预算 8、最多 2 次视觉修改。

下载、参考渲染和批处理 manifest 记录了数据 revision、对象 ID、输入哈希、模型配置、逐任务终止状态、模型调用数和 token 数。该次运行记录的代码状态为 `d936205e4902afc88681e008165294f22297522d`，工作树标记为 dirty。
