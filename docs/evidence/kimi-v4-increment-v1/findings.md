# Pilot findings

测试了10条Kimi监督能否改善v4学生。数值为17/28→18/28，所有预测变化仅是offset的parts编号变化，不能建立视觉能力或教师语义的因果增益。单seed、7资产、任务曝光变化是主要混杂。

保留数据接入结果与原始失败：纯offset仍1/5，组合来源存在1例原正确定位退化。后续优先补可信offset/parts对照和资产覆盖，避免只复制当前clean/axis配比。用户仅授权一组最小对照，因此不自动补实验，不改变训练框架或冻结Test。

result-to-claim: partial / medium，same-family provisional；完整experiment-audit未运行。
