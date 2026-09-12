import json

import numpy as np
import pytest

from turnitover.core.actions import ActuateJoint, Detent
from turnitover.oracle.evidence import ExposureThresholds, pixel_signal
from turnitover.oracle.supervision import action_examples
from turnitover.oracle.witnesses import Witness, next_witness, plan, recipe, witness_grid, zero_detent


def test_negative_limits_restore_at_upper_limit_and_count_cost():
    joints = {"a": (-1., 0.), "b": (0., 1.)}
    current = Witness("front", "a", Detent.ZERO)
    target = Witness("oblique_fr", "b", Detent.LIMIT)
    actions = recipe(current, target, joints)
    assert len(actions) == 3
    assert actions[0] == ActuateJoint("a", Detent.LIMIT)
    for action in actions:
        current = next_witness(current, action, joints)
    assert current == target
    with pytest.raises(ValueError, match="free reset"):
        zero_detent((-.3, 1.))


def test_planner_budget_and_state_match_each_observation():
    joints = {"a": (-1., 0.), "b": (0., 1.)}
    grid = witness_grid(["front", "oblique_fr", "back"], joints)
    exposures = [{0} if w.joint == "a" else {1} if w.joint == "b" else set() for w in grid]
    for mode in ("oracle", "fixed", "random"):
        for budget in range(1, 14):
            trajectory = plan(grid, exposures, joints, budget, mode=mode, seed=42)
            assert len(trajectory) <= budget
            state = Witness("front")
            for action, index in trajectory:
                state = next_witness(state, action, joints)
                assert state == grid[index]
    trajectory = plan(grid, exposures, joints, 4)
    assert set().union(*(exposures[i] for _, i in trajectory)) == {0, 1}
    assert trajectory == plan(grid, exposures, joints, 4)


def test_pixel_signal_detects_difference_but_not_identity():
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = a.copy()
    b[:4, :4] = 255
    assert not pixel_signal(a, a)["exposed"]
    assert pixel_signal(a, b)["exposed"]
    assert not pixel_signal(a, b, ExposureThresholds(image_l1=.5))["exposed"]


def test_action_examples_never_include_current_or_future_images():
    context = dict(task="verify", parts=["door"], joints=["hinge"], views=["front"],
                   runtime_properties=[], action_types=["request_view", "actuate_joint"],
                   max_triangles=5000, max_draw_calls=32)
    history = [dict(step=0, action=dict(type="request_view", view_id="front"), image_ref="past.png", payload={}),
               dict(step=1, action=dict(type="actuate_joint", joint_id="hinge", detent="limit"), image_ref="future.png", payload={})]
    rows = list(action_examples("case", context, "reference.png", history, 2))
    assert rows[0]["images"] == ["reference.png"]
    assert rows[1]["images"] == ["reference.png", "past.png"]
    assert "future.png" not in json.dumps(rows)
    assert "past.png" not in rows[0]["messages"][0]["content"]
    assert json.loads(rows[1]["messages"][1]["content"])["action"] == history[1]["action"]
