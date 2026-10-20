# Round 2 来源容量审计

- downloaded: 400
- structural_candidates: 142
- structural_rejected: 258
- base_objects: 38
- fresh_verified_queue: 142
- currently_certified_objects: 38
- currently_certified_pairs_samples: 76
- new_automatic_pass: 0
- new_human_pass: 0
- quality_unverified_object_upper_bound: 180

仅为来源合格且未重复的待渲染候选上界；尚无新增自动或人工通过样本。来源许可与修订沿用既有官方记录。完整拒绝原因及保护集合见 private/capacity_audit.json。

统计口径：400个下载档案中含原有160个，已全部按旧来源排除；真正新增240个中142个结构通过、98个结构拒绝。上方 structural_rejected=258 是400个处理档案中的非候选总数（160个旧来源排除 + 98个新增结构拒绝），不代表新增拒绝258个。142个来源候选仍需原渲染/checker/真实性与真实人工审核门禁。
