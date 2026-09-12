# 关节资产库调研与接入建议

核查日期：2026-09-12。建议用 ReplicaCAD Interactive 验证第一条真实网格转换链路，
以 PartNet-Mobility 作为规模化数据主来源，GAPartNet 作为部件标注补充。
以下下载与许可状态来自官方发布页；ReplicaCAD 已有转换实测，其他数据源的接入判断尚待实测。

| 资产库 | 对本项目的价值 | 获取与许可 | 接入判断 |
| --- | --- | --- | --- |
| PartNet-Mobility | 约 2K 个带运动标注与渲染材质的对象，适合主要训练数据 | 官方 Hugging Face 需要登录、申请并获管理员批准；限定非商业研究与教育，另需遵守 ShapeNet 条款 | 优先规模化来源；URDF/网格导入后仍需质量过滤 |
| GAPartNet | 1,166 个对象、27 类、8,489 个部件实例，提供语义与姿态标注 | 官方仓库声明 CC BY-NC 4.0，数据经表单申请并附额外条款 | 对定位任务很有价值；与其他资产库的对象重合须按上游身份去重 |
| ReplicaCAD Interactive | 包含可交互家具与门的 URDF，覆盖旋转/平移关节 | 官方 CC BY 4.0；公开 Hugging Face 文件，本次已下载关节子集 | 适合转换试验和工程验收；独立物体数量不足以支持大规模泛化结论 |
| AKB-48 | 论文报告 48 类、2,037 个真实扫描对象，有结构与物理属性 | 作者主页源码提供 Google Drive 下载入口；本次未确认资产的明确授权文本与完整包内容 | 保留为后续真实扫描来源，不把论文中的属性描述当作已验证可用数据 |

来源：[PartNet-Mobility 官方数据卡](https://huggingface.co/datasets/sapien-sim/PartNetMobility)、
[GAPartNet 官方仓库](https://github.com/PKU-EPIC/GAPartNet)、
[GAPartNet 论文](https://arxiv.org/abs/2211.05272)、
[ReplicaCAD 官方说明](https://aihabitat.org/datasets/replica_cad/)、
[AKB-48 论文](https://arxiv.org/abs/2202.08432)、
[AKB-48 作者下载页源码](https://github.com/liuliu66/AKB-48/blob/gh-pages/download.html)。

## 本次实际获取

运行 `python scripts/download_replicacad.py`，从官方
[ReplicaCAD Hugging Face 仓库](https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset)
下载 `urdf/` 与原始 README，固定 revision 为 `3e8c7fe5759f64bfcbc3882f9cdf6de97f82a06d`。

- 本地目录：`data/assets/replicacad-3e8c7fe/`（不进入 Git）。
- 42 个文件，共 40,266,703 字节，含 12 份 URDF。
- 下载逐文件核对上游 Git blob SHA1 / LFS SHA256 与大小，并另存 SHA256。
- `acquisition.json` 记录版本、来源、许可、文件清单、关节与网格引用；本次未发现缺失引用。
- 12 份 URDF 包含 static/dynamic 变体与门变体，不能当作 12 个独立原物体。
- 后续已转换六个资产族并完成 102 状态审计；详见 [真实网格 benchmark](mesh-benchmark.md)。

复现时默认目录必须不存在；可用 `--out` 指定新目录。无需安装 Habitat，无需账号。
PartNet-Mobility 和 GAPartNet 尚未申请或下载，也未通过第三方镜像绕过官方访问流程。

## 从原始资产到当前仓库的缺口

当前已扩展 `AssetSpec` 的嵌入网格和关节固定旋转，并实现窄范围 URDF 转换器。
下面是其接入验收要求；碰撞体/物理属性和完整纹理保真仍待后续实现：

1. 保留 link/joint 树、固定关节、关节 origin 的平移与旋转、轴、限位及 mesh scale。
2. 支持一个 link 对应多个 visual，明确视觉网格与碰撞体分开保存。
3. 处理 GLB 内部节点变换和材质；浏览器必须离线加载所有依赖。
4. 在 rest、上下限与中间状态核对前向运动学，避免仅比较静止截图。
5. 按原始对象划分资产组；保留转换器版本、上游 ID、哈希和许可。

实际清点的 cabinet 文件出现 `scale="1, 1, 1"` 的逗号形式、负的关节下限和零上限，
且多份 URDF 明确标记 dummy inertia。转换器需兼容这些数值格式，不能把 `limit` 默认理解为“打开”，
也不能直接用文件中的惯量开展物理正确性评测。这些是本次读取原始文件的发现。

第一批建议选 cabinet、chest_of_drawers、fridge 各一个非 dynamic 版本做转换验收。
三类资产只用于验证链路；正式数据集的类别数、对象数和腐蚀覆盖在接入 PartNet-Mobility 后确定。
