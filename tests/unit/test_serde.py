import pytest

from turnitover.core.actions import Action, Detent
from turnitover.core.sample import Framing, Sample
from turnitover.core.serde import from_dict, to_dict


def test_sample_wire_format_and_round_trip(sample):
    d = to_dict(sample)
    assert d["trajectory"][1]["action"]["__type__"] == "ActuateJoint"
    assert d["trajectory"][1]["action"]["detent"] == "limit"
    back = from_dict(Sample, d)
    assert back == sample
    assert back.trajectory[1].action.detent is Detent.LIMIT
    # Legacy records predate the dataset mapping but use the same action encoding.
    d.pop("dataset")
    d["schema_version"] = 1
    assert from_dict(Sample, d).dataset == {}


@pytest.mark.parametrize("data", [
    {"__type__": "UnknownAction", "view_id": "front"},
    {"view_id": "front"},
    None,
])
def test_bad_action_tags_never_become_valid_actions(data):
    with pytest.raises(ValueError, match="tag"):
        from_dict(Action, data)


@pytest.mark.parametrize("center", [[1, 2], [1, 2, 3, 4]])
def test_fixed_vectors_reject_missing_or_extra_entries(center):
    with pytest.raises(ValueError):
        from_dict(Framing, {"center": center, "radius": 1})


def test_optional_dataclasses_preserve_tags_and_nullability():
    assert from_dict(Framing | None, None) is None
    assert from_dict(Framing | None, {"center": [1, 2, 3], "radius": 1}) == Framing((1, 2, 3), 1)
    with pytest.raises(ValueError, match="type tag"):
        from_dict(Framing | None, {"__type__": "Unknown", "center": [1, 2, 3], "radius": 1})
    with pytest.raises(ValueError, match="Null"):
        from_dict(Framing, None)
