"""One shared budget/validation loop for active and baseline verification."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from turnitover.core.actions import Observation, action_cost
from turnitover.core.serde import to_dict
from turnitover.models.client import ModelError
from turnitover.policy.loop import ImageSink, SessionLike, execute_observation
from turnitover.verifier.contracts import Context, DecisionError, Verdict, parse_decision, uncertain

log = logging.getLogger("turnitover.verifier")


class VerificationPolicy(Protocol):
    def decide(self, context: Context, history: Sequence[Observation], remaining: int) -> dict: ...


@dataclass(frozen=True)
class Episode:
    verdict: Verdict
    termination: str
    trajectory: tuple[Observation, ...]
    spent: int
    error_type: str | None = None
    attempted_action: dict | None = None
    error_message: str | None = None


def run_verification(session: SessionLike, policy: VerificationPolicy, context: Context, budget: int,
                     image_sink: ImageSink, observation_sink: Callable[[Observation], None] = lambda obs: None) -> Episode:
    if type(budget) is not int or budget < 0:
        raise ValueError("Observation budget must be a nonnegative integer")
    history: list[Observation] = []
    spent = 0
    # At most K observations + one final decision. Failed actions also spend their attempted cost.
    for _ in range(budget + 1):
        remaining = budget - spent
        try:
            raw = policy.decide(context, tuple(history), remaining)
            decision = parse_decision(raw, context, {o.step for o in history}, final_only=remaining == 0)
        except DecisionError as exc:
            return Episode(uncertain("Judge returned an invalid decision."), "invalid_decision", tuple(history), spent,
                           "DecisionError", error_message=str(exc))
        except ModelError:
            return Episode(uncertain("Judge API call failed or returned an incomplete response."), "model_error", tuple(history), spent, "ModelError")
        if isinstance(decision, Verdict):
            return Episode(decision, "budget_exhausted" if remaining == 0 else "judge_finished", tuple(history), spent)
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
                           tuple(history), spent, type(exc).__name__, to_dict(decision))
        history.append(obs)
        observation_sink(obs)
    raise AssertionError("Verification loop did not terminate")
