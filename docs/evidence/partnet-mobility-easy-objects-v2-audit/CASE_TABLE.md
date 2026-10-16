# 逐案例核对（含私有答案，不用于盲审）

| 资产 | 图板行 | case_id | Gold | 人审 verdict / type | joint / parts | 证据状态 | 四字段 | 后续候选 |
|---|---:|---|---|---|---|---|---|---|
| PM_102016 | 2 | eor-24f1b5b48879 | clean | pass / None | None /  | 未逐状态引用 | 正确 | 高置信度候选 |
| PM_102016 | 3 | eor-3da26638899e | part_offset | fail / part_offset | None / link_0 | rest,mid,limit | 正确 | 高置信度候选 |
| PM_102016 | 4 | eor-68c7fb547080 | joint_axis | fail / joint_axis | joint_0 / link_0 | mid,limit | 正确 | 高置信度候选 |
| PM_102379 | 2 | eor-42443d5a386a | part_offset | fail / part_offset | None / link_0 | rest,mid,limit | 正确 | 高置信度候选 |
| PM_102379 | 3 | eor-472c836efefa | clean | pass / None | None /  | 未逐状态引用 | 正确 | 高置信度候选 |
| PM_102379 | 4 | eor-6a5cc960eafc | joint_axis | fail / joint_axis | joint_0 / link_0 | limit | 正确 | 语义或上下文待确认 |
| PM_10697 | 2 | eor-57a7ca1300ca | joint_axis | fail / joint_axis | joint_1 / link_1 | mid,limit | 正确 | 高置信度候选 |
| PM_10697 | 3 | eor-72321a1e08e9 | part_offset | fail / part_offset | None / link_1 | rest | 正确 | 高置信度候选 |
| PM_10697 | 4 | eor-79926ed20b6c | clean | pass / None | None /  | 未逐状态引用 | 正确 | 高置信度候选 |
| PM_12594 | 2 | eor-49a23abdda64 | part_offset | pass / None | None /  | 未逐状态引用 | explicit_error | 误判保留 |
| PM_12594 | 3 | eor-9d3143231c41 | joint_axis | fail / joint_axis | joint_1 / link_1 | mid,limit | 正确 | 高置信度候选 |
| PM_12594 | 4 | eor-e4f19bdb3db0 | clean | pass / None | None /  | 未逐状态引用 | 正确 | 高置信度候选 |
| PM_19836 | 2 | eor-a76c53e335c6 | clean | pass / None | None /  | 未逐状态引用 | 正确 | 高置信度候选 |
| PM_19836 | 3 | eor-d32e9efba9de | joint_axis | fail / joint_axis | joint_0 / link_0 | limit | 正确 | 高置信度候选 |
| PM_19836 | 4 | eor-f67bca1f30ef | part_offset | fail / part_offset | None / link_0 | rest,limit | 正确 | 高置信度候选 |
| PM_48686 | 2 | eor-0407b54342e0 | clean | pass / None | None /  | 未逐状态引用 | 正确 | 高置信度候选 |
| PM_48686 | 3 | eor-0b44adc5d35c | part_offset | fail / part_offset | None / link_0 | rest,mid,limit | 正确 | 高置信度候选 |
| PM_48686 | 4 | eor-e8c7469405c6 | joint_axis | fail / joint_axis | joint_0 / link_0 | mid,limit | 正确 | 高置信度候选 |
| PM_5088 | 2 | eor-38d4df7a4778 | part_offset | pass / None | None /  | rest,mid,limit | explicit_error | 误判保留 |
| PM_5088 | 3 | eor-95a0f2bbba88 | clean | pass / None | None /  | rest,mid,limit | 正确 | 高置信度候选 |
| PM_5088 | 4 | eor-d5ba7047c72f | joint_axis | fail / joint_axis | joint_0 / link_0 | mid,limit | 正确 | 高置信度候选 |
| PM_8867 | 2 | eor-6b8564f7552a | part_offset | fail / part_offset | None / link_1 | rest | 正确 | 语义或上下文待确认 |
| PM_8867 | 3 | eor-85eff6ea6b76 | clean | pass / None | None /  | 未逐状态引用 | 正确 | 语义或上下文待确认 |
| PM_8867 | 4 | eor-a6c870bf59bf | joint_axis | fail / joint_axis | joint_1 / link_1 | mid,limit | 正确 | 语义或上下文待确认 |
