import numpy as np
import pytest

from turnitover.corruptions import get_corruption, registered_ids


@pytest.mark.parametrize("defect_id", registered_ids())
def test_deterministic_given_seed(toy_program, defect_id):
    c = get_corruption(defect_id)
    assert c.applicable(toy_program)
    p1, s1 = c.apply(toy_program, np.random.default_rng(42))
    p2, s2 = c.apply(toy_program, np.random.default_rng(42))
    p3, s3 = c.apply(toy_program, np.random.default_rng(43))
    assert p1.source == p2.source and s1 == s2
    assert p1.source != p3.source or s1 != s3
    assert s1.defect_id == defect_id
    assert 0.0 <= s1.severity <= 1.0
    assert set(s1.parts) <= set(toy_program.spec.part_ids)


@pytest.mark.parametrize("defect_id", registered_ids())
def test_only_declared_parts_change(toy_program, defect_id):
    c = get_corruption(defect_id)
    new, spec = c.apply(toy_program, np.random.default_rng(1))
    old_parts = {p.id: p for p in toy_program.spec.parts}
    new_parts = {p.id: p for p in new.spec.parts}
    for pid in old_parts:
        if pid not in spec.parts:
            assert old_parts[pid] == new_parts[pid]
    if defect_id == "kinematics.joint_axis":
        # rest pose unchanged: parts identical, only a joint axis differs
        assert old_parts == new_parts
        changed = [j for j in new.spec.joints if j != toy_program.spec.joint(j.id)]
        assert len(changed) == 1 and changed[0].part in spec.parts
        assert abs(np.linalg.norm(changed[0].axis) - 1.0) < 1e-6
