"""Crash-resumable teacher/environment loop with an atomic episode ledger."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Sequence

from turnitover.core.actions import Observation, action_cost
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.serde import to_dict
from turnitover.models.client import ModelError
from turnitover.policy.loop import SessionLike, execute_observation
from turnitover.render.session import HarnessError
from turnitover.teacher.policy import TeacherPolicy
from turnitover.telemetry import now_iso
from turnitover.verifier.contracts import Context, DecisionError, Verdict, parse_decision
from turnitover.verifier.loop import MAX_DECISION_ATTEMPTS, _validation_feedback


def _observation(record: dict, context: Context) -> Observation:
    action = parse_decision({"action": record["decision"]}, context, set())
    return Observation(record["step"], action, record.get("image_ref"), record.get("payload"),
                       record.get("elapsed_ms", 0.0))


def _totals(calls: Sequence[dict]) -> dict:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for call in calls:
        usage = call.get("usage") or {}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        totals["input_tokens"] += input_tokens if type(input_tokens) is int else 0
        totals["output_tokens"] += output_tokens if type(output_tokens) is int else 0
        total = usage.get("total_tokens")
        totals["total_tokens"] += total if type(total) is int else (
            (input_tokens if type(input_tokens) is int else 0)
            + (output_tokens if type(output_tokens) is int else 0))
    return totals


def _checkpoint(path: Path, state: dict, policy: TeacherPolicy) -> None:
    state["model_calls"] = policy.calls
    state["token_usage"] = _totals(policy.calls)
    state["updated_at"] = now_iso()
    write_json_atomic(path, state)


def _replay(session: SessionLike, output: Path, context: Context, records: Sequence[dict]) -> list[Observation]:
    history = []
    for record in records:
        expected = output / record["image_ref"] if record.get("image_ref") else None

        def sink(step, png):
            if expected is None or hashlib.sha256(png).hexdigest() != hashlib.sha256(expected.read_bytes()).hexdigest():
                raise ValueError("Resume replay image differs from the saved causal observation")
            return record["image_ref"]

        saved = _observation(record, context)
        execute_observation(session, saved.action, saved.step, sink)
        history.append(saved)
    return history


def run_teacher_episode(session: SessionLike, policy: TeacherPolicy, context: Context, budget: int,
                        output: Path, *, episode_id: str, resume: bool = False,
                        max_model_calls: int | None = None,
                        decision_attempts: int = MAX_DECISION_ATTEMPTS,
                        data_identity: dict | None = None,
                        initial_actions: Sequence[dict] = ()) -> dict:
    """Run one episode. A pending action is durable before environment execution."""
    if type(budget) is not int or budget < 0:
        raise ValueError("Teacher observation budget must be a nonnegative integer")
    if type(decision_attempts) is not int or decision_attempts < 1:
        raise ValueError("Teacher decision attempts must be positive")
    if resume and initial_actions:
        raise ValueError("Fixed initial actions cannot be added while resuming")
    result_path = output / "episode.json"
    if resume:
        if not result_path.is_file():
            raise ValueError("No saved episode is available to resume")
        state = json.loads(result_path.read_text())
        if state.get("status") != "interrupted":
            raise ValueError("Only an interrupted episode can be resumed")
        if state["episode_id"] != episode_id or state["budget"] != budget:
            raise ValueError("Resume identity or budget differs from the saved episode")
        if data_identity is not None and state.get("data_identity") != data_identity:
            raise ValueError("Resume data identity differs from the saved episode")
        policy.calls = list(state.get("model_calls", []))
        history = _replay(session, output, context, state["trajectory"])
    else:
        if result_path.exists():
            raise ValueError("Teacher output directory must be new")
        output.mkdir(parents=True, exist_ok=True)
        (output / "observations").mkdir()
        state = {"schema_version": 1, "episode_id": episode_id, "status": "running",
                 "created_at": now_iso(), "budget": budget, "spent": 0,
                 "context": dataclasses.asdict(context), "teacher": policy.teacher.public(),
                 "trajectory": [], "decisions": [], "pending_action": None,
                 "model_calls": [], "token_usage": _totals([]),
                 "budget_exhausted": False, "teacher_verdict_missing": True}
        if data_identity is not None:
            state["data_identity"] = data_identity
        history = []
        _checkpoint(result_path, state, policy)

    def image_sink(step: int, png: bytes) -> str:
        relative = f"observations/{step:03d}.png"
        (output / relative).write_bytes(png)
        return relative

    def execute_pending() -> None:
        raw = state["pending_action"]
        action = parse_decision({"action": raw}, context, {o.step for o in history})
        observation = execute_observation(session, action, len(history), image_sink)
        record = to_dict(observation)
        record["decision"] = raw
        record["decision_source"] = state.pop("pending_action_source", "teacher")
        state["trajectory"].append(record)
        history.append(observation)
        state["pending_action"] = None
        _checkpoint(result_path, state, policy)

    try:
        if not resume:
            for raw in initial_actions:
                if state["spent"] >= budget:
                    raise ValueError("Fixed initial actions exceed the observation budget")
                parse_decision({"action": raw}, context, {o.step for o in history})
                state["spent"] += 1
                state["pending_action"] = raw
                state["pending_action_source"] = "fixed_protocol"
                _checkpoint(result_path, state, policy)
                execute_pending()
        if state.get("pending_action") is not None:
            try:
                execute_pending()
            except HarnessError as exc:
                state.update(status="error", termination="observation_error",
                             error_type=type(exc).__name__, error_message=str(exc),
                             attempted_action=state.pop("pending_action"), finished_at=now_iso())
                return _finish(result_path, state, policy)
        while True:
            remaining = budget - state["spent"]
            if max_model_calls is not None and len(policy.calls) >= max_model_calls:
                state.update(status="complete", termination="model_call_limit",
                             budget_exhausted=remaining == 0, teacher_verdict_missing=True,
                             finished_at=now_iso())
                break
            feedback = None
            for attempt in range(decision_attempts):
                if max_model_calls is not None and len(policy.calls) >= max_model_calls:
                    state.update(status="complete", termination="model_call_limit",
                                 budget_exhausted=remaining == 0, teacher_verdict_missing=True,
                                 finished_at=now_iso())
                    return _finish(result_path, state, policy)
                raw = None
                try:
                    raw = policy.decide(context, tuple(history), remaining, feedback)
                    decision = parse_decision(raw, context, {o.step for o in history}, final_only=remaining == 0)
                    state["decisions"].append({"call": len(policy.calls) - 1, "remaining": remaining,
                                               "valid": True, "decision": raw})
                    _checkpoint(result_path, state, policy)
                    break
                except DecisionError as exc:
                    state["decisions"].append({"call": len(policy.calls) - 1, "remaining": remaining,
                                               "valid": False, "decision": raw, "error": str(exc)})
                    _checkpoint(result_path, state, policy)
                    if attempt == decision_attempts - 1:
                        exhausted = remaining == 0
                        state.update(status="complete" if exhausted else "error",
                                     termination="budget_exhausted" if exhausted else "invalid_decision",
                                     budget_exhausted=exhausted, teacher_verdict_missing=True,
                                     error_type="DecisionError", error_message=str(exc),
                                     finished_at=now_iso())
                        return _finish(result_path, state, policy)
                    feedback = _validation_feedback(exc, raw, context, history, remaining)
                except ModelError as exc:
                    state.update(status="error", termination="model_error", error_type="ModelError",
                                 error_message=str(exc), budget_exhausted=remaining == 0,
                                 teacher_verdict_missing=True, finished_at=now_iso())
                    return _finish(result_path, state, policy)
            if isinstance(decision, Verdict):
                exhausted = remaining == 0
                state.update(status="complete", termination="budget_exhausted" if exhausted else "teacher_finished",
                             verdict=dataclasses.asdict(decision), budget_exhausted=exhausted,
                             teacher_verdict_missing=False, finished_at=now_iso())
                break
            state["spent"] += action_cost(decision)
            state["pending_action"] = raw["action"]
            _checkpoint(result_path, state, policy)
            try:
                execute_pending()
            except HarnessError as exc:
                state.update(status="error", termination="observation_error",
                             error_type=type(exc).__name__, error_message=str(exc),
                             attempted_action=state.pop("pending_action"), finished_at=now_iso())
                return _finish(result_path, state, policy)
    except Exception as exc:
        state.update(status="interrupted", termination="interrupted", error_type=type(exc).__name__,
                     error_message=str(exc))
        _checkpoint(result_path, state, policy)
        raise
    return _finish(result_path, state, policy)


def _finish(path: Path, state: dict, policy: TeacherPolicy) -> dict:
    state["budget_exhausted"] = bool(state.get("budget_exhausted") or state["spent"] >= state["budget"])
    state["teacher_verdict_missing"] = "verdict" not in state
    _checkpoint(path, state, policy)
    return state
