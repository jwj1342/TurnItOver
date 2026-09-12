"""Privileged evaluation, deliberately absent from the policy context and decision loop."""
from turnitover.checkers.evidence import collect_evidence, resolve_states
from turnitover.checkers.geometry import GeometryAlignmentChecker
from turnitover.checkers.runtime_budget import RuntimeBudgetChecker
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession


def audit(candidate, reference, render, views, max_triangles, max_draw_calls):
    try:
        with ObservationSession(render, views) as session:
            info = session.load(reference, framing=None)
            states = resolve_states(("rest", "joint:*:limit"), tuple(info.joints))
            gold = collect_evidence(session, reference.sha, states)
            session.load(candidate, framing=info.framing)
            observed = collect_evidence(session, candidate.sha, states)
            checks = (GeometryAlignmentChecker().check(observed, gold),
                      RuntimeBudgetChecker(max_triangles, max_draw_calls).check(observed, None))
            return {"status": "complete", "reference_sha": reference.sha, "checks": [to_dict(c) for c in checks],
                    "privileged": True, "used_by_judge": False}
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "privileged": True, "used_by_judge": False}
