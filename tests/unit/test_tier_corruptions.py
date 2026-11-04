import math
import numpy as np
import pytest
from turnitover.core.program import ObjectProgram
from turnitover.corruptions import get_corruption
from turnitover.corruptions.kinematics import coordinate_axis

def test_axis_permutation_swaps_to_another_signed_coordinate_axis(toy_program):
    corruption = get_corruption("kinematics.joint_axis", mode="axis_permutation")
    seen = set()
    for seed in range(200):
        program, label = corruption.apply(toy_program, np.random.default_rng(seed))
        joint_id = label.params["joint"]
        old, new = toy_program.spec.joint(joint_id), program.spec.joint(joint_id)
        source, target = coordinate_axis(old.axis), coordinate_axis(new.axis)
        assert target is not None and target[0] != source[0]
        assert (new.sign, new.anchor, new.limits, new.type) == (old.sign, old.anchor, old.limits, old.type)
        assert label.params["mode"] == "axis_permutation" and label.params["angle_deg"] == 90.0
        assert (label.params["from_index"], label.params["to_index"], label.params["to_sign"]) == (
            source[0], target[0], target[1])
        assert all(math.copysign(1.0, x) > 0 for x in new.axis if x == 0.0)  # no "-0.0" literal in the code
        assert program.spec.parts == toy_program.spec.parts
        assert [j for j in program.spec.joints if j != toy_program.spec.joint(j.id)] == [new]
        seen.add((source[0], target))
    # both signs and both other axes occur for the same source axis, drawn from the explicit generator
    assert {(1, (0, 1)), (1, (0, -1)), (1, (2, 1)), (1, (2, -1))} <= seen
    first = corruption.apply(toy_program, np.random.default_rng(5))
    assert first[0].source == corruption.apply(toy_program, np.random.default_rng(5))[0].source


def test_axis_permutation_skips_joints_whose_axis_is_not_a_coordinate_axis(toy_spec):
    tilted = toy_spec.replace_joint("door", axis=(0.0, math.sqrt(0.5), math.sqrt(0.5)))
    only_tilted = type(tilted)(tilted.asset_id, tilted.category, tilted.parts,
                               tuple(j for j in tilted.joints if j.id == "door"), tilted.materials)
    corruption = get_corruption("kinematics.joint_axis", mode="axis_permutation")
    assert not corruption.applicable(ObjectProgram.from_spec(only_tilted))
    with pytest.raises(ValueError):
        corruption.apply(ObjectProgram.from_spec(only_tilted), np.random.default_rng(0))
    mixed = ObjectProgram.from_spec(tilted)  # drawers still qualify; the tilted door is never picked
    assert corruption.applicable(mixed)
    assert all(corruption.apply(mixed, np.random.default_rng(s))[1].params["joint"] != "door" for s in range(50))


def test_random_angle_stays_a_separate_stratum_and_leaves_the_coordinate_axes(toy_program):
    corruption = get_corruption("kinematics.joint_axis")
    assert corruption.mode == "axis_random_angle"
    _, label = corruption.apply(toy_program, np.random.default_rng(1))
    assert label.params["mode"] == "axis_random_angle"
    assert all(coordinate_axis(corruption.apply(toy_program, np.random.default_rng(s))[1].label["axis"]) is None
               for s in range(20))
    with pytest.raises(ValueError):
        get_corruption("kinematics.joint_axis", mode="nonsense")


# ---- (b) axis negation == sign negation -----------------------------------------------------------------------

def test_direction_flip_negates_axis_and_keeps_sign(toy_program):
    corruption = get_corruption("kinematics.direction_flip")
    for seed in range(20):
        program, label = corruption.apply(toy_program, np.random.default_rng(seed))
        old, new = toy_program.spec.joint(label.params["joint"]), program.spec.joint(label.params["joint"])
        assert new.sign == old.sign
        assert new.axis == tuple(-x for x in old.axis) and (new.anchor, new.limits) == (old.anchor, old.limits)
        assert label.params["mode"] == "axis_negation"
        assert all(math.copysign(1.0, x) > 0 for x in new.axis if x == 0.0)
