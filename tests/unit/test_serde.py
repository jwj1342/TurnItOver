from turnitover.core.actions import ActuateJoint, Detent, Observation, QueryRuntime, RequestView, RuntimeProperty, Stop, ViewId
from turnitover.core.sample import SCHEMA_VERSION, CheckResult, CorruptionSpec, Framing, Invariant, Sample
from turnitover.core.serde import from_dict, to_dict


def _sample() -> Sample:
    return Sample(
        schema_version=SCHEMA_VERSION, sample_id="toy-0", run_id="t", idx=0, seed=7, asset_id="toy", source_id="toy",
        program_sha="abc", program_source="export default 1", reference_sha="def",
        corruptions=(CorruptionSpec("structure.part_offset", ("door",), 0.5, {"m": 1.0}, {"part": "door"}),),
        framing=Framing((0.0, 0.4, 0.0), 0.6),
        trajectory=(
            Observation(0, RequestView(ViewId("front")), "obs/00.png", {"render_ms": 1.0}, 2.0),
            Observation(1, ActuateJoint("door", Detent.LIMIT), "obs/01.png", None, 2.0),
            Observation(2, QueryRuntime(RuntimeProperty.STATS), None, {"draw_calls": 3}, 1.0),
            Observation(3, Stop(), None, None, 0.0),
        ),
        checks=(CheckResult("runtime.budget", True, (Invariant("triangles", 12.0, 100.0, True),), 1.0),),
        compile_ok=True, error=None, timing_ms={"load": 3.0}, host="h", created_at="2026-09-11T00:00:00+00:00",
    )


def test_sample_round_trip():
    s = _sample()
    d = to_dict(s)
    assert d["trajectory"][1]["action"]["__type__"] == "ActuateJoint"
    assert from_dict(Sample, d) == s


def test_enum_and_none():
    s = _sample()
    d = to_dict(s)
    assert d["trajectory"][1]["action"]["detent"] == "limit"
    back = from_dict(Sample, d)
    assert back.trajectory[1].action.detent is Detent.LIMIT
    assert back.trajectory[3].action == Stop()
