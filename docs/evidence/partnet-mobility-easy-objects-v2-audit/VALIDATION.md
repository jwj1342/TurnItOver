# Audit validation

`PYTHONPATH=. python docs/evidence/partnet-mobility-easy-objects-v2-audit/reproduce.py`：exit 0。

校验了 24 个 case ID 集合一致、24 个保存程序的 SHA、对应参考 SHA、单 corruption
字段隔离、全部父产物文件哈希、17 个第一轮原包文件哈希，以及审核输入未被本轮改写。
13 个输入文件的 SHA 保存在 summary.json；独立 reviewer 也核对了其审阅时的输入哈希。

```bash
python -m pytest tests/unit/test_obvious_defect_adjudication.py \
  tests/unit/test_easy_object_review.py::test_single_corruption_isolation_preserves_motion_mechanism \
  tests/unit/test_easy_object_review.py::test_large_motion_difference_cannot_override_invisible_rest_offset -q
git diff --check
```

结果：4 项测试通过；diff whitespace 检查通过。没有声称整套测试或新 Chromium 重跑通过。
此次只核对既有 Chromium 证据，未生成新数据、调用 Kimi、导出训练集或训练模型。
