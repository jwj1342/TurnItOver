# 三案例完整执行记录核对

本目录是2026-10-21对既有实验的只读核对；不重新请求模型API，不修改原实验程序、图像、诊断、台账或评分。结论与范围见 [新的审查报告](../../docs/api-pipeline-record-review.md)。先前终审中断记录仍保留，不能将其日期状态改写为已完成。

- `check_records.py` / `checks.json` / `replay.log`：258项断言通过，真实Chromium重放22张原观察PNG，全部逐字节一致；六组逐轮/最终质量、诊断和几何见证重算一致。绑定原始API prompt、图像及响应，SSE独立组装匹配。运行时统计仅排除无法逐次复现的 `last_render_ms`，其余原统计相同。
- `execution-matrix.json`：各组初始/最终源码SHA、Verifier状态与独立Checker、调用与观察成本。一次API生成基础程序，另外两例是在基础程序上受控注入，不能描述为三次独立API生成。
- `repair-feedback-check.json`：两次修复调用只携带公开任务、模型诊断及其实际引用的观察；门轴补丁真实应用，干净程序收到stop后未改码。
- `cited-motion-check.json`：在真实浏览器中重放引用动作后导出的世界坐标几何，仅用于审计；c00 Fixed引用step2到3的抽屉位移为(0,0,+0.3)m，证实其“-X轴错误”解释错误。c02门几何及c01偏移同列供复查。
- `reviewer-trace/`：新代理独立脚本、核对数字及模型请求元数据；same-family/provisional，不是外部API供应商真实性证明。
- `replay-first-timing-check.log` 与 `replay-first-prompt-parser.log` 保留审计脚本初版对动态计时、多行提示解析的两次失败；修正仅涉及本目录审计脚本，未改变实验数据，也未新增API尝试。

从仓库根目录运行下面的离线核对；本脚本只重新写入本目录的 `checks.json`，不覆盖实验产物：

```bash
PYTHONPATH="$PWD" PLAYWRIGHT_BROWSERS_PATH=../TurnItOver/.cache/ms-playwright \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../TurnItOver/.venv-local/bin/python \
results/2026-10-21-api-record-audit/check_records.py
```

需要先解包 [原运行包](../2026-10-21-api-pipeline/README.md)，准备兼容Python/Playwright和npm依赖；保持归档的web/dist哈希，原始输出目录须齐全。使用新浏览器或驱动时PNG字节不保证相同，不能通过删除图像比对断言伪称复现。

观察K=4不包含两组共享的公开参考图、精确任务及候选部件/关节标识。Active额外模型调用、Fixed未覆盖第二关节、一次恢复的额外调用均需单独报告；Checker通过只是独立于模型的有限状态几何/runtime判定。`citations_exist`只检验引用编号存在，`geometric_witness_consistent_findings`是finding级“至少一个引用有几何不一致”的代理，并不保证每条引用或文字解释正确，也不覆盖未报告缺陷。
