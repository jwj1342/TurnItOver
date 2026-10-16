import pytest

from turnitover.oracle.motion_diagnosis import validate_target


@pytest.mark.parametrize("target", [
    {"defect_type": "axis", "joint": "joint_0", "part": "link_0", "evidence_steps": [2, 3]},
    {"defect_type": "offset", "joint": None, "part": "link_1", "evidence_steps": [0]},
    {"defect_type": "clean", "joint": None, "part": None, "evidence_steps": list(range(8))},
    {"defect_type": "uncertain", "joint": None, "part": None, "evidence_steps": []},
])
def test_motion_diagnosis_targets(target):
    validate_target(target)


def test_uncertain_cannot_leak_localization():
    with pytest.raises(ValueError, match="cannot localize"):
        validate_target({"defect_type": "uncertain", "joint": "joint_0", "part": None,
                         "evidence_steps": []})


def test_offset_cannot_claim_joint():
    with pytest.raises(ValueError, match="only part"):
        validate_target({"defect_type": "offset", "joint": "joint_0", "part": "link_0",
                         "evidence_steps": [0]})
