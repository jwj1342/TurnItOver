"""One shared budget/validation loop for active and baseline verification."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Callable, Protocol, Sequence

from turnitover.core.actions import Observation, action_cost, action_key
from turnitover.core.serde import to_dict
from turnitover.models.client import ModelError
from turnitover.policy.loop import ImageSink, SessionLike, execute_observation
from turnitover.verifier.contracts import Context, DecisionError, Verdict, parse_decision, uncertain

log = logging.getLogger("turnitover.verifier")
MAX_DECISION_ATTEMPTS = 3


class VerificationPolicy(Protocol):
    def decide(self, context: Context, history: Sequence[Observation], remaining: int,
               validation_feedback: str | None = None) -> dict: ...


@dataclass(frozen=True)
class Episode:
    verdict: Verdict
    termination: str
    trajectory: tuple[Observation, ...]
    spent: int
    error_type: str | None = None
    attempted_action: dict | None = None
    error_message: str | None = None
    action_decisions: tuple[dict, ...] = ()


def run_verification(session: SessionLike, policy: VerificationPolicy, context: Context, budget: int,
                     image_sink: ImageSink, observation_sink: Callable[[Observation], None] = lambda obs: None) -> Episode:
    if type(budget) is not int or budget < 0:
        raise ValueError("Observation budget must be a nonnegative integer")
    history: list[Observation] = []
    spent = 0
    action_decisions: list[dict] = []
    # At most K observations + one final decision. Failed actions also spend their attempted cost.
    for _ in range(budget + 1):
        remaining = budget - spent
        validation_feedback = None
        for attempt in range(MAX_DECISION_ATTEMPTS):
            raw = None
            try:
                raw = policy.decide(context, tuple(history), remaining, validation_feedback)
                decision = parse_decision(raw, context, {o.step for o in history}, final_only=remaining == 0)
                if not isinstance(decision, Verdict):
                    key = action_key(decision)
                    action_decisions.append({"legal": True, "action_key": key,
                                             "repeated": any(item.get("action_key") == key
                                                             for item in action_decisions)})
                break
            except DecisionError as exc:
                if isinstance(raw, dict) and "action" in raw:
                    action_decisions.append({"legal": False, "error": str(exc)})
                if attempt == MAX_DECISION_ATTEMPTS - 1:
                    return Episode(uncertain("Judge returned an invalid decision."), "invalid_decision",
                                   tuple(history), spent, "DecisionError", error_message=str(exc),
                                   action_decisions=tuple(action_decisions))
                validation_feedback = _validation_feedback(exc, raw, context, history, remaining)
            except ModelError:
                return Episode(uncertain("Judge API call failed or returned an incomplete response."), "model_error",
                               tuple(history), spent, "ModelError", action_decisions=tuple(action_decisions))
        if isinstance(decision, Verdict):
            return Episode(decision, "budget_exhausted" if remaining == 0 else "judge_finished", tuple(history), spent,
                           action_decisions=tuple(action_decisions))
        spent += action_cost(decision)
        log.info("verify.observe step=%s action=%s spent=%s/%s", len(history), type(decision).__name__, spent, budget)
        # Browser failures are distinguished from invalid judgments by the outer runner.
        try:
            obs = execute_observation(session, decision, len(history), image_sink)
        except Exception as exc:
            from turnitover.render.session import HarnessError

            if not isinstance(exc, HarnessError):
                raise
            return Episode(uncertain("Candidate failed while executing an observation."), "observation_error",
                           tuple(history), spent, type(exc).__name__, to_dict(decision),
                           action_decisions=tuple(action_decisions))
        history.append(obs)
        observation_sink(obs)
    raise AssertionError("Verification loop did not terminate")


def complete_final_readout(policy: VerificationPolicy, context: Context, episode: Episode) -> tuple[Episode, dict]:
    """Diagnose an acquired prefix after acquisition ended without a verdict."""
    if episode.termination in {"budget_exhausted", "judge_finished"}:
        return episode, {"supplemental": False, "status": "already_complete", "calls": 0}
    feedback = None
    for attempt in range(MAX_DECISION_ATTEMPTS):
        raw = None
        try:
            raw = policy.decide(context, episode.trajectory, 0, feedback)
            verdict = parse_decision(raw, context, {o.step for o in episode.trajectory}, final_only=True)
            return replace(episode, verdict=verdict), {
                "supplemental": True, "status": "complete", "calls": attempt + 1,
                "acquisition_termination": episode.termination,
            }
        except DecisionError as exc:
            if attempt + 1 < MAX_DECISION_ATTEMPTS:
                feedback = _validation_feedback(exc, raw, context, episode.trajectory, 0)
                continue
            error = exc
        except ModelError as exc:
            error = exc
        return episode, {"supplemental": True, "status": "failed", "calls": attempt + 1,
                         "acquisition_termination": episode.termination,
                         "error_type": type(error).__name__, "error": str(error)}
    raise AssertionError("Final readout did not terminate")


def _validation_feedback(error: DecisionError, raw: dict | None, context: Context,
                         history: Sequence[Observation], remaining: int) -> str:
    details = {
        "validation_error": str(error),
        "rejected_decision": raw,
        "legal_values": {
            "action_types": context.action_types,
            "view_ids": context.views,
            "joint_ids": context.joints,
            "detents": ("zero", "mid", "limit"),
            "runtime_properties": context.runtime_properties,
            "observed_steps": tuple(o.step for o in history),
        },
        "remaining_observation_budget": remaining,
    }
    return "The previous decision failed validation. Correct it using these details:\n" + json.dumps(details, ensure_ascii=False)
