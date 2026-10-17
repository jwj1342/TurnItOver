# 文档索引

本仓库包含实验执行 Pipeline 与 Chatbox 轨迹 Viewer。本文档树按“使用方式、稳定接口、架构、实验记录”组织，不再以里程碑编号作为主要导航。

## 使用指南

- [Getting started](getting-started.md)：安装环境，启动 Viewer 或实验 Pipeline。
- [CLI reference](cli-reference.md)：当前受支持的五个入口、参数、退出码和输出目录行为。
- [Configuration](configuration.md)：`.env` 字段、默认值、校验范围和覆盖关系。
- [Data and artifacts](data-and-artifacts.md)：输入数据、参考资产、本地 run 产物和隐私边界。

## 架构

- [Milestone 1–5 阶段总结](coder-verifier-pipeline-milestone.md)：按项目演进顺序概览各阶段；稳定接口仍以专题架构文档为准。
- [Active-view Verifier](architecture/active-view-verifier.md)：V1/V2 动作空间、ObservationRuntime、camera harness 和 observation budget。
- [Iterative Pipeline](architecture/iterative-pipeline.md)：CodeAgent、render gate、Verifier 和 visual revision 闭环。

## 实验记录

- [Experiment documentation policy](experiments/README.md)：正式实验结果必须记录的 provenance，以及说明性产物示例。

## 当前实现边界

| 子系统 | 当前状态 |
|---|---|
| Chatbox `session.json + resources/` Viewer | 已实现 |
| Runtime render check | 已实现 |
| CodeAgent 与 runtime repair | 已实现 |
| Active-view Verifier | 已实现 |
| V1 relative action space | 默认可用 |
| V2 pose-grid runtime | 已实现；要求 candidate camera harness |
| M5 iterative pipeline | 已实现 |
| Streamlit 回放 Pipeline native trajectory | 已实现；支持版本化 agent session 与本地 run |

旧 Codex JSONL parser 和对应 smoke-test 文档已从当前文档接口移除。
