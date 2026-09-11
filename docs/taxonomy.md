# Defect taxonomy

GENERATED from `turnitover/taxonomy/defects.yaml` by `python -m turnitover render-docs`. Do not edit by hand. Source table: RP.md, section 方法/缺陷分类与数据构造.


| id | 层级 | 缺陷 | 注入方式 | 可见性类别 | 说明 | 标签形式 | 已实现 |
|---|---|---|---|---|---|---|---|
| `geometry.thin_sheet` | geometry | 实体替换为共面薄片 | 挤出体改为极薄面片 | view_dependent | 掠射视角可见，正视不可见 | part_set | no |
| `geometry.open_volume` | geometry | 未封闭体积 | 删除端盖或开放边界 | probe_required | 通常不可见，需几何探针 | part_set | no |
| `geometry.self_intersection` | geometry | 自相交 | 扰动轮廓使其自交 | view_dependent | 部分可见 | part_set | no |
| `geometry.global_scale` | geometry | 全局尺度错误 | 整体等比缩放 | context_dependent | 无参照物时不可见 | scalar | no |
| `geometry.anisotropic_scale` | geometry | 各向异性比例错误 | 单轴缩放 | static_visible | 单视角可见 | part_scalar | no |
| `geometry.missing_part` | geometry | 部件缺失 | 删除组件 | static_visible | 单视角常可见 | part_id | no |
| `geometry.duplicate_part` | geometry | 部件冗余 | 复制并微移 | view_dependent | 部分可见 | part_id | no |
| `geometry.missing_bevel` | geometry | 倒角丢失 | 关闭 bevel | view_dependent | 近景可见 | part_set | no |
| `structure.wrong_parent` | structure | 父节点挂错 | 改变层级归属 | animation_required | 静态不可见，动画时暴露 | hierarchy_edge | no |
| `structure.pivot_offset` | structure | 枢轴位置偏移 | 平移 pivot | animation_required | 静态不可见，动画时暴露 | part_vector | no |
| `structure.part_offset` | structure | 部件位置偏移 | 平移部件 | static_visible | 单视角常可见 | part_vector | yes |
| `structure.part_rotation` | structure | 部件朝向错误 | 旋转部件 | static_visible | 单视角常可见 | part_rotation | no |
| `structure.symmetry_break` | structure | 对称性破坏 | 单侧扰动 | view_dependent | 正视可见 | part_pair | no |
| `structure.interpenetration` | structure | 部件互穿 | 平移使其相交 | view_dependent | 部分可见 | part_pair | no |
| `structure.floating` | structure | 悬空无接触 | 抬升部件 | view_dependent | 部分可见 | part_id | no |
| `kinematics.joint_axis` | kinematics | 关节轴方向错误 | 旋转轴向量 | animation_required | 静态不可见，需动画序列 | part_axis | yes |
| `kinematics.joint_type` | kinematics | 关节类型错误 | 旋转副与移动副互换 | animation_required | 静态不可见，需动画序列 | part_class | no |
| `kinematics.joint_range` | kinematics | 行程范围错误 | 放大或缩小限位 | animation_required | 静态不可见，需动画末态 | part_interval | no |
| `kinematics.joint_anchor` | kinematics | 关节锚点错误 | 平移锚点 | animation_required | 静态不可见，需动画序列 | part_position | no |
| `kinematics.direction_flip` | kinematics | 运动方向反向 | 参数取负 | animation_required | 静态不可见，需动画序列 | part_bool | no |
| `material.basecolor_as_rough` | material | 基础色误接为粗糙度 | 通道错接 | static_visible | 可见，高光跟随图案 | material_id | no |
| `material.metalness` | material | 金属度错误 | metalness 翻转 | static_visible | 可见 | material_scalar | no |
| `material.tiling_scale` | material | 贴图平铺比例错误 | 改变 repeat | static_visible | 可见 | material_id | no |
| `material.emissive` | material | 自发光误设 | emissive 非零 | static_visible | 可见 | material_id | no |
| `material.normal_map` | material | 法线强度或方向错误 | 取反或放大 | view_dependent | 近景可见 | material_id | no |
| `runtime.triangle_budget` | runtime | 三角面超预算 | 提高细分段数 | probe_required | 不可见，需统计探针 | scalar | yes |
| `runtime.draw_calls` | runtime | draw call 过多 | 拆散网格或取消实例化 | probe_required | 不可见，需统计探针 | scalar | no |
| `runtime.missing_dispose` | runtime | 资源未释放 | 移除 dispose 调用 | probe_required | 不可见，需运行时探针 | bool | no |
| `runtime.tick_missing_or_phase` | runtime | 动画函数缺失或相位错误 | 删除或偏移 tick | temporal_required | 静态不可见，需时序 | bool | no |

## Visibility by layer

| layer | static_visible | view_dependent | animation_required | probe_required | temporal_required | context_dependent |
|---|---|---|---|---|---|---|
| geometry | 2 | 4 | 0 | 1 | 0 | 1 |
| structure | 2 | 3 | 2 | 0 | 0 | 0 |
| kinematics | 0 | 0 | 5 | 0 | 0 | 0 |
| material | 4 | 1 | 0 | 0 | 0 | 0 |
| runtime | 0 | 0 | 0 | 3 | 1 | 0 |
