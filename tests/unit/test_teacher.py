import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from turnitover.render.session import HarnessError
from turnitover.teacher.audit import audit_episode
from turnitover.teacher.episode import run_teacher_episode
from turnitover.teacher.policy import TeacherPolicy
from turnitover.teacher.providers import KimiPricing, KimiTeacher, MockTeacher
from turnitover.models.config import ModelConfig
from turnitover.verifier.contracts import Context


@pytest.fixture
def context():
    return Context("inspect", ("base", "door"), ("hinge",), ("front", "side"),
                   ("joint_state",), ("request_view", "actuate_joint", "query_runtime"), 100, 10)


def action(view="front"):
    return {"action": {"type": "request_view", "view_id": view}}


def verdict():
    return {"verdict": {"status": "uncertain", "confidence": 0.0, "findings": [],
                        "summary": "insufficient", "limitations": ["bounded evidence"]}}


class Session:
    def __init__(self, fail=None):
        self.fail = fail
        self.actions = []

    def request_view(self, view):
        self.actions.append(view)
        if self.fail:
            exc, self.fail = self.fail, None
            raise exc
        return SimpleNamespace(png=f"png:{view}".encode(), render_ms=1, camera={"view": view})

    def actuate(self, joint, detent):
        self.actions.append((joint, detent))
        return SimpleNamespace(png=f"png:{joint}:{detent}".encode(), render_ms=1, value=1)

    def query_runtime(self, prop):
        self.actions.append(prop)
        return {"joint": "state"}


def make_policy(tmp_path, decisions):
    reference = tmp_path / "reference-00.png"
    reference.write_bytes(b"reference")
    return TeacherPolicy(MockTeacher(decisions), tmp_path, [reference])


def test_mock_episode_closes_budget_and_audits_visible_request(tmp_path, context):
    policy = make_policy(tmp_path, [action(), verdict()])
    episode = run_teacher_episode(Session(), policy, context, 1, tmp_path, episode_id="case")
    audit = audit_episode(tmp_path)
    request = json.loads((tmp_path / "model_calls/001/request.json").read_text())
    assert episode["termination"] == "budget_exhausted" and episode["spent"] == 1
    assert episode["budget_exhausted"] and not episode["teacher_verdict_missing"]
    assert request["remaining"] == 0 and len(request["history"]) == 1
    assert request["stop_required"] is True
    assert request["image_order"][-1] == {"image": 1, "kind": "observation", "step": 0}
    assert set(request) == {"context", "remaining", "stop_required", "image_order", "history"}
    assert audit["pipeline_ok"] and audit["teacher_actions_legal"]
    assert audit["teacher_outputs_legal"] and audit["teacher_verdict_present"]


def test_teacher_can_stop_before_budget_is_exhausted(tmp_path, context):
    episode = run_teacher_episode(Session(), make_policy(tmp_path, [verdict()]), context, 2,
                                  tmp_path, episode_id="case")
    request = json.loads((tmp_path / "model_calls/000/request.json").read_text())
    assert episode["termination"] == "teacher_finished" and episode["spent"] == 0
    assert not episode["budget_exhausted"] and not episode["teacher_verdict_missing"]
    assert request["remaining"] == 2 and request["stop_required"] is False


def test_budget_exhaustion_without_valid_verdict_is_explicit_and_not_fabricated(tmp_path, context):
    episode = run_teacher_episode(Session(), make_policy(tmp_path, [action(), action()]), context, 1,
                                  tmp_path, episode_id="case", decision_attempts=1)
    audit = audit_episode(tmp_path)
    assert episode["status"] == "complete" and episode["termination"] == "budget_exhausted"
    assert episode["budget_exhausted"] and episode["teacher_verdict_missing"]
    assert "verdict" not in episode
    assert audit["pipeline_ok"] and not audit["teacher_outputs_legal"]
    assert not audit["teacher_verdict_present"] and audit["teacher_verdict_missing"]


@pytest.mark.parametrize("responses", [
    [action("secret")] * 3,
    ["not-json"] * 3,
])
def test_invalid_or_unparseable_actions_are_audited_without_budget_charge(tmp_path, context, responses):
    episode = run_teacher_episode(Session(), make_policy(tmp_path, responses), context, 1,
                                  tmp_path, episode_id="case")
    assert episode["termination"] == "invalid_decision" and episode["spent"] == 0
    assert episode["teacher_verdict_missing"] and "verdict" not in episode
    assert len(episode["decisions"]) == 3 and not any(item["valid"] for item in episode["decisions"])
    audit = audit_episode(tmp_path)
    assert audit["pipeline_ok"] and not audit["teacher_outputs_legal"]


def test_execution_failure_is_charged_and_saved(tmp_path, context):
    episode = run_teacher_episode(Session(HarnessError("runtime", "broken")),
                                  make_policy(tmp_path, [action()]), context, 1,
                                  tmp_path, episode_id="case")
    assert episode["termination"] == "observation_error" and episode["spent"] == 1
    assert episode["attempted_action"]["view_id"] == "front"


def test_resume_executes_pending_action_without_second_teacher_selection(tmp_path, context):
    class SecondActionInterrupt(Session):
        def request_view(self, view):
            if len(self.actions) == 1:
                self.actions.append(view)
                raise RuntimeError("interrupt")
            return super().request_view(view)

    policy = make_policy(tmp_path, [action(), action("side"), verdict()])
    with pytest.raises(RuntimeError):
        run_teacher_episode(SecondActionInterrupt(), policy, context, 2,
                            tmp_path, episode_id="case")
    resumed_session = Session()
    resumed = run_teacher_episode(resumed_session, policy, context, 2, tmp_path,
                                  episode_id="case", resume=True)
    audit = audit_episode(tmp_path)
    assert resumed["status"] == "complete" and resumed["spent"] == 2
    assert len(resumed["model_calls"]) == 3 and len(resumed["trajectory"]) == 2
    assert resumed_session.actions == ["front", "side"]
    assert audit["pipeline_ok"] and audit["teacher_verdict_present"]


def test_kimi_requires_allowlisted_model_permission_and_bounded_cost():
    cfg = ModelConfig("teacher", "openai_compatible", "k3", "https://api.example/v1", "key", 100)
    price = KimiPricing(1, 2, "2026-10-14", "official pricing page")
    with pytest.raises(ValueError, match="permission"):
        KimiTeacher(cfg, price, permission_confirmed=False, estimated_input_tokens=100)
    teacher = KimiTeacher(cfg, price, permission_confirmed=True, estimated_input_tokens=100)
    assert teacher.public()["max_real_requests"] == 1
