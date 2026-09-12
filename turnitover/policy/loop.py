"""run_observation_loop: drive a policy against a session under an observation budget.

Storage-agnostic: images go to ``image_sink(step, png) -> image_ref``.
"""
from __future__ import annotations

import time
from typing import Callable, Protocol

from turnitover.core.actions import (
    ActuateJoint, EmitDiagnosis, Observation, QueryRuntime, RequestView, Stop, action_cost,
)
from turnitover.policy.base import JudgePolicy, PolicyContext

ImageSink = Callable[[int, bytes], str]


class SessionLike(Protocol):
    """The subset of ObservationSession the loop needs (lets tests use a fake)."""

    def request_view(self, view_id: str): ...

    def actuate(self, joint_id: str, detent): ...

    def query_runtime(self, prop) -> dict: ...


def execute_observation(session: SessionLike, action: Action, step: int, image_sink: ImageSink) -> Observation:
    """Shared action execution for dataset scripts and online verification. No hidden probes."""
    t0 = time.perf_counter()
    image_ref = None
    payload = None
    match action:
        case RequestView(view_id=v):
            res = session.request_view(v)
            image_ref = image_sink(step, res.png)
            payload = {"render_ms": res.render_ms, "camera": res.camera}
        case ActuateJoint(joint_id=j, detent=d):
            res = session.actuate(j, d)
            image_ref = image_sink(step, res.png)
            payload = {"value": res.value, "render_ms": res.render_ms}
        case QueryRuntime(property=p):
            payload = session.query_runtime(p)
        case EmitDiagnosis() | Stop():
            pass
        case _:
            raise TypeError("Unsupported observation action")
    return Observation(step, action, image_ref, payload, (time.perf_counter() - t0) * 1000)


def run_observation_loop(
    session: SessionLike,
    policy: JudgePolicy,
    context: PolicyContext,
    image_sink: ImageSink,
    max_steps: int = 64,
) -> tuple[Observation, ...]:
    policy.reset(context)
    history: list[Observation] = []
    spent = 0
    for step in range(max_steps):
        action = policy.act(history)
        if context.budget is not None and spent + action_cost(action) > context.budget:
            action = Stop()
        spent += action_cost(action)
        history.append(execute_observation(session, action, step, image_sink))
        if isinstance(action, (Stop, EmitDiagnosis)):
            break
    return tuple(history)
