import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from turnitover.core.actions import Observation, QueryRuntime, RequestView, RuntimeProperty, ViewId
from turnitover.models.client import ModelError, ModelResult
from turnitover.render.session import HarnessError
from turnitover.verifier.contracts import Context, DecisionError, parse_decision, parse_json
from turnitover.verifier.loop import run_verification
from turnitover.verifier.policies import RuntimePolicy, VisionPolicy


@pytest.fixture
def context():
    return Context("Inspect cabinet", ("door",), ("hinge",), ("front",), ("stats",),
                   ("request_view", "actuate_joint", "query_runtime"), 5000, 32)


def final(status="uncertain", findings=None):
    return {"verdict": {"status": status, "confidence": 0.8, "findings": findings or [],
                        "summary": "Review completed", "limitations": ["Hidden geometry not verified"]}}


def finding():
    return {"defect_id": "runtime.triangle_budget", "parts": [], "severity": 0.5, "confidence": 1,
            "evidence_steps": [0], "description": "Triangle budget exceeded", "suggested_fix": "Reduce triangles"}


class Session:
    def __init__(self):
        self.actions = []

    def request_view(self, view):
        self.actions.append(view)
        return SimpleNamespace(png=b"png", render_ms=1, camera={})

    def query_runtime(self, prop):
        self.actions.append(prop)
        return {"triangles_scene": 7000, "draw_calls": 9}

    def actuate(self, joint, detent):
        self.actions.append((joint, detent))
        return SimpleNamespace(png=b"png", render_ms=1, value=1)


class Policy:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.remaining = []

    def decide(self, context, history, remaining):
        self.remaining.append(remaining)
        return next(self.decisions)


def test_budget_allows_final_verdict_after_last_observation(context):
    policy = Policy([{"action": {"type": "request_view", "view_id": "front"}}, final("pass")])
    session = Session()
    result = run_verification(session, policy, context, 1, lambda i, png: f"{i}.png")
    assert policy.remaining == [1, 0]
    assert result.spent == 1 and len(result.trajectory) == 1
    assert result.verdict.status == "pass" and result.termination == "budget_exhausted"


def test_zero_budget_cannot_act_or_pass(context):
    for decision in ({"action": {"type": "request_view", "view_id": "front"}}, final("pass")):
        session = Session()
        result = run_verification(session, Policy([decision]), context, 0, lambda *a: "x")
        assert not session.actions and result.verdict.status == "uncertain"
        assert result.termination == "invalid_decision"


@pytest.mark.parametrize("action", [
    {"type": "request_view", "view_id": "secret"},
    {"type": "actuate_joint", "joint_id": "absent", "detent": "limit"},
    {"type": "actuate_joint", "joint_id": "hinge", "detent": []},
    {"type": "query_runtime", "property": "export_geometry"},
    {"type": "request_view", "view_id": "front", "extra": True},
])
def test_invalid_actions_do_not_reach_browser(context, action):
    session = Session()
    result = run_verification(session, Policy([{"action": action}]), context, 1, lambda *a: "x")
    assert not session.actions and result.termination == "invalid_decision"


def test_action_ablation_is_enforced(context):
    context = dataclasses.replace(context, action_types=("request_view",))
    with pytest.raises(DecisionError):
        parse_decision({"action": {"type": "query_runtime", "property": "stats"}}, context, set())


@pytest.mark.parametrize("key,value", [("confidence", float("nan")), ("severity", True),
    ("parts", ["missing"]), ("evidence_steps", [99]), ("evidence_steps", []),
    ("defect_id", "made.up"), ("description", "")])
def test_findings_validate_ranges_parts_and_actual_evidence(context, key, value):
    f = finding()
    f[key] = value
    with pytest.raises(DecisionError):
        parse_decision(final("fail", [f]), context, {0})


def test_valid_evidence_linked_finding(context):
    assert parse_decision(final("fail", [finding()]), context, {0}).findings[0].evidence_steps == (0,)


def test_qwen_pass_with_findings_is_rejected_with_explanation(context):
    # Regression from the first real local Qwen trials: correct finding, contradictory pass.
    policy = Policy([{"action": {"type": "query_runtime", "property": "stats"}}, final("pass", [finding()])])
    result = run_verification(Session(), policy, context, 1, lambda *a: "unused")
    assert result.verdict.status == "uncertain" and result.termination == "invalid_decision"
    assert "no findings" in result.error_message


def test_duplicate_json_keys_rejected():
    with pytest.raises(DecisionError):
        parse_json('{"verdict": {}, "verdict": {}}')


def test_model_error_preserves_previous_observations(context):
    class FailingPolicy:
        def decide(self, context, history, remaining):
            if history:
                raise ModelError("network")
            return {"action": {"type": "request_view", "view_id": "front"}}
    result = run_verification(Session(), FailingPolicy(), context, 2, lambda *a: "image.png")
    assert result.termination == "model_error" and result.spent == 1 and len(result.trajectory) == 1


def test_failed_action_is_charged_and_identified(context):
    class Broken(Session):
        def request_view(self, view):
            raise HarnessError("runtime", "oops")
    result = run_verification(Broken(), Policy([{"action": {"type": "request_view", "view_id": "front"}}]),
                              context, 2, lambda *a: "x")
    assert result.spent == 1 and result.termination == "observation_error"
    assert result.attempted_action["view_id"] == "front" and result.trajectory == ()


def test_runtime_baseline_cannot_claim_visual_pass(context):
    observed = [Observation(0, QueryRuntime(RuntimeProperty.STATS), None, {"triangles_scene": 100, "draw_calls": 9}, 1)]
    assert RuntimePolicy().decide(context, observed, 0)["verdict"]["status"] == "uncertain"
    observed[0].payload["triangles_scene"] = 7000
    decision = RuntimePolicy().decide(context, observed, 0)
    assert parse_decision(decision, context, {0}).status == "fail"


def test_vision_receives_ordered_images_without_gold_or_source(tmp_path, context):
    calls = []
    def model(prompt, images):
        calls.append((prompt, images))
        return ModelResult(json.dumps(final()), "mock", {}, "completed", 1)
    reference = tmp_path / "reference.png"
    policy = VisionPolicy(model, tmp_path, [reference])
    history = [Observation(0, RequestView(ViewId("front")), "observations/000.png", {}, 1)]
    policy.decide(context, history, 0)
    assert calls[0][1] == [reference, tmp_path / "observations/000.png"]
    assert '"action":' not in calls[0][0].split("Observed task data:")[0]
    data = json.loads(calls[0][0].split("Observed task data:\n")[1])
    assert data["image_order"][1]["step"] == 0
    assert set(data) == {"context", "remaining", "taxonomy_ids", "image_order", "history"}
    assert (tmp_path / "model_calls/000/response.txt").exists()


def test_fixed_baseline_uses_one_final_model_call(tmp_path, context):
    calls = []
    def model(prompt, images):
        calls.append(prompt)
        return ModelResult(json.dumps(final()), "mock", {}, "completed", 1)
    policy = VisionPolicy(model, tmp_path, [], mode="fixed")
    result = run_verification(Session(), policy, context, 2, lambda i, p: f"{i}.png")
    assert len(calls) == 1 and result.spent == 2


def test_random_plan_is_seeded(tmp_path, context):
    a = VisionPolicy(None, tmp_path, [], "random", 11)
    b = VisionPolicy(None, tmp_path, [], "random", 11)
    assert a.decide(context, [], 8) == b.decide(context, [], 8)
    assert a._plan == b._plan
