# Level 0 extreme joint-axis review

This is a **pending-human-review** visual bundle: five independent Train assets, each with a clean candidate and one 90-degree `kinematics.joint_axis` candidate. No `part_offset`, teacher call, SFT run, or Test row is included.

Open `boards/` for rest/mid/limit comparisons. `inputs/` is the concise four-tile input for a 2B diagnosis: reference rest/limit on the first row, candidate rest/limit on the second. All tiles are native Chromium 256x256 renders, and reference/candidate use the same frozen normal camera and state. IDs are opaque and the public manifest omits labels.

For each row in `human_review.jsonl`, decide without opening `private/`, enlarging the image, or using hierarchy: whether the candidate is correct; whether it is one-glance obvious; and one sentence describing the normal versus candidate motion direction. Keep every row pending until that review is complete. The private folder holds injection truth, specs, strict-screen diagnostics, and independent checker outputs.
