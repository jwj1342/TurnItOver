from dataclasses import dataclass

from turnitover.core.actions import ActuateJoint, Detent, QueryRuntime, RequestView, RuntimeProperty, Stop, ViewId
from turnitover.policy.base import PolicyContext
from turnitover.policy.loop import run_observation_loop
from turnitover.policy.scripted import ScriptedPolicy


@dataclass
class _Res:
    png: bytes = b"png"
    render_ms: float = 1.0
    camera: dict = None
    value: float = 0.0


class FakeSession:
    def __init__(self):
        self.calls = []

    def request_view(self, view_id):
        self.calls.append(("view", view_id))
        return _Res(camera={})

    def actuate(self, joint_id, detent):
        self.calls.append(("joint", joint_id, detent))
        return _Res(value=0.35)

    def query_runtime(self, prop):
        self.calls.append(("runtime", prop))
        return {"property": prop.value}


SCRIPT = [RequestView(ViewId("front")), ActuateJoint("d", Detent.LIMIT), QueryRuntime(RuntimeProperty.STATS), RequestView(ViewId("top")), Stop()]


def test_uncapped_runs_full_script():
    s = FakeSession()
    images = {}
    traj = run_observation_loop(s, ScriptedPolicy(SCRIPT), PolicyContext(("d",), (ViewId("front"),), None), lambda i, b: images.setdefault(f"obs/{i:02d}.png", b))
    assert len(traj) == 5 and isinstance(traj[-1].action, Stop)
    assert len(s.calls) == 4 and set(images) == {"obs/00.png", "obs/01.png", "obs/03.png"}
    assert traj[2].payload == {"property": "stats"}


def test_budget_forces_stop():
    s = FakeSession()
    traj = run_observation_loop(s, ScriptedPolicy(SCRIPT), PolicyContext(("d",), (), 2), lambda i, b: "x")
    assert len(s.calls) == 2
    assert isinstance(traj[-1].action, Stop) and len(traj) == 3
