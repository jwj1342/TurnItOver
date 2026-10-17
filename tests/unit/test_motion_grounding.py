import numpy as np
import pytest
from turnitover.oracle.motion_grounding import compare_observations, visible_comparison


def square(x):
    a=np.zeros((64,64,3),dtype=np.uint8);a[20:40,x:x+16]=255
    return a


def compare(pairs, states=None):
    states=states or [{'step':0,'view':'front','actual_value':0}, {'step':1,'view':'front','actual_value':1}]
    return visible_comparison(compare_observations(pairs,states,0),states,0)


def test_offset_can_have_temporal_residual_without_axis():
    c=compare([(square(10),square(18)),(square(25),square(33))])
    assert c['static_rest']=='different'
    assert c['motion_change']=='different'  # Never label this a rotation-axis measurement.


def test_identical_motion_and_missing_motion():
    assert compare([(square(10),square(10)),(square(25),square(25))])['motion_change']=='matching'
    assert compare([(square(10),square(10))],[{'step':4,'view':'front','actual_value':0}])['motion_change']=='unobserved'


def test_order_independent_public_step_identity():
    p=[(square(10),square(10)),(square(25),square(35))]
    s=[{'step':0,'view':'front','actual_value':0},{'step':1,'view':'front','actual_value':1}]
    assert compare(p,s)['motion_pairs']==compare(p[::-1],s[::-1])['motion_pairs']==[[0,1]]


def test_cannot_compare_different_camera_as_motion():
    s=[{'step':0,'view':'front','actual_value':0},{'step':1,'view':'back','actual_value':1}]
    assert compare([(square(10),square(10)),(square(25),square(35))],s)['motion_change']=='unobserved'


def test_mismatched_dimensions_rejected():
    with pytest.raises(ValueError):
        compare([(square(10),square(10)[:32])],[{'step':0,'view':'front','actual_value':0}])
