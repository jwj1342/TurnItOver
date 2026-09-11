"""collect_evidence: drive the session to export what checkers need (rest + joint detents)."""
from __future__ import annotations

from typing import Sequence

from turnitover.checkers.base import Evidence
from turnitover.core.actions import REST_STATE, Detent, RuntimeProperty, joint_state_key
from turnitover.render.session import ObservationSession


def resolve_states(patterns: Sequence[str], joint_ids: Sequence[str]) -> tuple[str, ...]:
    """Expand ["rest", "joint:*:limit"] into concrete state keys for the loaded joints."""
    out: list[str] = []
    for p in patterns:
        if p == REST_STATE:
            out.append(REST_STATE)
            continue
        kind, jid, detent = p.split(":")
        assert kind == "joint", p
        d = Detent(detent)
        for j in (joint_ids if jid == "*" else [jid]):
            out.append(joint_state_key(j, d))
    return tuple(dict.fromkeys(out))


def collect_evidence(session: ObservationSession, program_sha: str, states: Sequence[str]) -> Evidence:
    info = session.info
    geometry: dict[str, dict] = {}
    for key in states:
        session.reset_joints()
        if key != REST_STATE:
            _, jid, detent = key.split(":")
            session.actuate(jid, Detent(detent))
        geometry[key] = session.export_geometry()
    session.reset_joints()
    stats = session.query_runtime(RuntimeProperty.STATS)
    hierarchy = session.query_runtime(RuntimeProperty.HIERARCHY)
    return Evidence(program_sha=program_sha, compile_ok=True, error=None, geometry=geometry, stats=stats, hierarchy=hierarchy)


def failed_evidence(program_sha: str, error: str) -> Evidence:
    return Evidence(program_sha=program_sha, compile_ok=False, error=error, geometry={}, stats={}, hierarchy={})
