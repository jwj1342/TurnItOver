# 默认测试必要性审查

本轮基于上一轮清理后的工作区逐项检查默认测试，依据是它能否捕获独立且实际重要的错误。
参数化用例保留不同错误路径的代表输入；不通过改成循环、改 marker 或跳过测试来压低计数。

## 删除与合并

| 原测试或断言 | 处理与依据 |
| --- | --- |
| `test_emit_deterministic` | 删除单独的自比较。三个 corruption 的测试已经重复生成程序并比较结果。 |
| `test_replace_changes_source` | 删除弱的单次源码不等检查；在上述三个 corruption 测试中同时确认源码和 SHA 都相对原程序变化，覆盖实际使用的生成路径。 |
| `test_enum_and_none` | 合入 Sample 往返测试，保留枚举编码和枚举对象身份检查。StrEnum 与字符串可以相等，所以身份检查不能仅用 dataclass 相等替代。 |
| `test_taxonomy_integrity` | 删除固定数量 29 和重复的类型、唯一性、label_form 检查；其他保留测试调用的加载器已经校验这些约束。 |
| `test_kinematics_all_animation_required` | 删除固定数量 5 和所有运动缺陷必须共享一个可见性分类的假设。具体标签由版本化 taxonomy 定义，不将当前标签表硬编码成额外通用规则。 |
| `test_valid_evidence_linked_finding` | 将 evidence_steps 断言合入已有 runtime 判定测试，仍校验完整合法 finding 的解析。 |
| serde 的两个未知标签案例 | 保留带 view_id 的未知标签，删除另一份未知标签；缺失标签和非法 null 仍分别测试。 |
| audit 的 startup/checker 错误参数 | 删除重复的通用错误传播案例；保留候选 HarnessError 被记录、候选内部 TypeError 传播、参考 HarnessError 传播，区分错误类型与发生位置。 |
| OpenRouter 请求格式参数 | 它当前和 openai_compatible 共用同一格式分支，保留后者作为代表；没有删除实际 provider 支持。如果将来引入专属请求逻辑，再添加对应案例。 |
| 快照报告的逐字段常量断言 | 保留与已归档完整 analysis.json 的相等比较、文件哈希和摘要篡改检查；删除完整比较后再次写死成本、调用数、成功数等字段的断言。 |
| storage 的 tar 存在断言 | 后续实际读取 tar 已经验证它存在，删除重复的 exists 检查。 |

同时将随机策略测试改为运行完整 episode，比较实际输出的动作序列、seed 复现性和动作集合，
不再访问 `_plan` 或调用生产代码的 `action_plan` 生成期望值。paired diagnosis 的隐私断言
读取结构化 prompt 数据，不再依赖说明文字中的特定短语。

共享 Sample 数据移到 `tests/conftest.py`，storage/dataset 不再导入另一个测试模块中的私有函数。
旧版 Sample 的默认字段兼容断言回归 serde 测试，避免混在名字声称“拒绝旧记录”的数据切分测试中。

## 保留测试对应的失败风险

下表覆盖本轮保留的全部 20 个默认测试模块。小而直接的测试不因行数少而删除；涉及不同阶段的
相似断言也不自动视为重复，例如数据生成预算与 verifier 预算属于两个独立循环。

| 模块（`tests/unit/`） | 保留原因 |
| --- | --- |
| `test_corruptions.py` | 三种真实 corruption 的可复现性、程序变化、标签范围、非目标零件不变及关节轴约束。 |
| `test_dataset.py` | 扰动不等于违规、标签证据有效性、资产层级和导入字段、隔离导出、坏样本隔离、跨 split 泄漏、单资产误切分、uncertain 按漏检计分。 |
| `test_loop.py` | 数据生成循环的完整动作执行、图片归属和预算强制停止。 |
| `test_media.py` | 实际视频帧数、奇数尺寸编码，以及渲染失败时不发布半成品。需要系统 FFmpeg，因此属于默认套件中的集成检查。 |
| `test_meshops.py` | IoU 基本边界、表面采样、解析距离、细分不变性、大三角形漏检、退化面，以及与独立暴力算法对照。 |
| `test_models.py` | 环境配置优先级和非执行解析、凭据不泄漏、四种请求格式和响应解析、实际 HTTP 适配路径、dry-run 不调用 API、生成输出歧义和截断。 |
| `test_oracle.py` | 负关节限位恢复成本、各预算下动作与状态一致、像素暴露阈值、训练输入不读取未来观测。 |
| `test_oracle_ambiguity.py` | 同一公开输入的频率权重、跨 split 隔离、图片内容保持和未来证据隔离。 |
| `test_oracle_diagnosis.py` | 未暴露的 gold 不变成 finding 或 pass，已暴露证据能定位到实际 step，成对图片顺序及公开数据边界。 |
| `test_oracle_evaluate.py` | 有效命中、误报、无效输出漏检、gold 不入模型，以及 single/paired/identical 条件的图片和计分差异。 |
| `test_oracle_training.py` | prompt token 不参与梯度，assistant loss 符合经验分布权重；本地无 torch 时跳过，不因此删除。 |
| `test_protocol.py` | 二进制 typed-array 解码的数值与索引，避免错误 endian、dtype 或 reshape。 |
| `test_repair.py` | 原子且无重叠的补丁、保留回退轨迹及失败成本、audit 故障传播、准备图片与实际请求一致、输入包篡改、不同 provider 的 token 计费。 |
| `test_research_snapshot.py` | 已归档测量文件完整性、从 episodes 重算研究报告、摘要与 episode 不一致时拒绝继续。 |
| `test_serde.py` | 合法 wire format 和往返类型、旧记录默认值、未知/缺失标签、非法 null、固定向量长度及 Optional 类型边界。 |
| `test_sharding.py` | 分片不重不漏、历史 seed 映射稳定且不同输入可区分、CLI 的两种 shard 表示。 |
| `test_storage.py` | 实际 tar/index/image/geometry 写入读取和多样本边界，不能由纯 serde 往返代替。 |
| `test_taxonomy.py` | 实现映射与标签表一致，生成文档与标签定义不漂移。 |
| `test_urdf.py` | 非单位、嵌套关节坐标系的 FK 数值，以及不支持的关节不能静默冻结。 |
| `test_verifier.py` | 最后一次观测后仍能判定、零预算、动作消融与非法动作、证据约束、真实模型矛盾输出、失败观测成本、基线局限、图片顺序、固定/随机策略及 audit 错误边界。 |

保留理由不等于证明测试已穷尽所有错误。未来功能删除、实验契约变化或出现新的重复覆盖时，
应重新检查相关测试；105 不是长期固定指标。

## 验证结果

- 默认套件：105 passed、1 skipped（未安装 torch）、12 browser 用例默认不运行；上一轮为 115 passed、1 skipped。
- 单独运行浏览器套件：12 passed。
- 删除 6 个测试函数、4 个重复参数案例；包含共享 fixture 在内，测试 Python 代码从 1,594 行减少到 1,546 行，净减 48 行。
- tests 通过 pyflakes，工作区 diff 通过空白检查。
- 六种一次性进程内错误注入均导致相关测试失败：取消 shuffle、打乱计划但实际仍按固定顺序执行、未知动作转成 Stop、向量静默截断、emitter 返回常量、丢弃 false negative。

错误注入只在临时子进程内运行，没有修改仓库实现或增加常驻测试框架。这是对所列错误的
敏感性验证，不是全仓 mutation coverage，也不是所有剩余测试都不可替代的证明。
