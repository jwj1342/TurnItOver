import json
from dataclasses import asdict

import pytest

from turnitover.core.actions import Observation, RequestView
from turnitover.models.client import ModelResult
from turnitover.oracle.supervision import action_examples
from turnitover.verifier.action_only import ActionOnlyPolicy
from turnitover.verifier.contracts import Context, DecisionError


CONTEXT = Context("Inspect candidate", ("body",), ("hinge",), ("front", "top"), (),
                  ("request_view", "actuate_joint"), 100, 10)


def result(value):
    return ModelResult(json.dumps(value), "fake", {}, "completed", 1.)


def test_online_prefix_matches_original_export_and_excludes_future(tmp_path):
    calls = []
    def action(prompt, images):
        calls.append((prompt, images))
        return result({"action": {"type": "request_view", "view_id": "top"}})
    policy = ActionOnlyPolicy(action, None, tmp_path, "reference.png")
    history = [Observation(0, RequestView("front"), "first.png", {"camera": {}, "render_ms": 999}, 20)]
    policy.decide(CONTEXT, history, 3)
    exported = list(action_examples("uid", asdict(CONTEXT), "reference.png", [
        {"step": 0, "action": {"type": "request_view", "view_id": "front"},
         "image_ref": "first.png", "payload": {"camera": {}}},
        {"step": 1, "action": {"type": "request_view", "view_id": "top"}}], 4))[-1]
    assert calls[0][0] == exported["messages"][0]["content"]
    assert calls[0][1] == [tmp_path / "reference.png", tmp_path / "first.png"]
    assert "render_ms" not in calls[0][0]


def test_acquisition_rejects_early_verdict_and_uses_frozen_final(tmp_path):
    action_calls, final_calls = [], []
    verdict = {"verdict": {"status": "uncertain", "confidence": 0., "findings": [],
                            "summary": "Insufficient evidence", "limitations": ["No motion"]}}
    def action(prompt, images):
        action_calls.append(prompt)
        return result(verdict)
    def final(prompt, images):
        final_calls.append(prompt)
        return result(verdict)
    policy = ActionOnlyPolicy(action, final, tmp_path, "reference.png")
    with pytest.raises(DecisionError, match="Action-only"):
        policy.decide(CONTEXT, (), 2)
    assert not final_calls
    assert policy.decide(CONTEXT, (), 0) == verdict
    assert len(action_calls) == len(final_calls) == 1


def test_schema_illegal_action_counts_attempt_but_not_legal(tmp_path):
    def action(prompt, images):
        return result({"action": {"type": "request_view", "view_id": "unavailable"}})
    policy = ActionOnlyPolicy(action, None, tmp_path, "reference.png")
    with pytest.raises(DecisionError, match="Unknown view"):
        policy.decide(CONTEXT, (), 2)
    assert policy.action_attempts == 1
    assert policy.legal_actions == 0
    assert policy.calls[0]["action_only_json"] is True


def test_truncated_response_is_saved_and_rejected(tmp_path):
    from turnitover.models.client import ModelError
    def action(prompt, images):
        return ModelResult('{"action":{"type":"request_view","view_id":"front"}}',
                           "fake", {}, "incomplete", 1.)
    policy = ActionOnlyPolicy(action, None, tmp_path, "reference.png")
    with pytest.raises(ModelError, match="incomplete"):
        policy.decide(CONTEXT, (), 2)
    assert (tmp_path / "action_calls/000/response.txt").exists()
    assert policy.action_attempts == 1 and policy.legal_actions == 0


def test_runtime_history_keeps_payload_without_inventing_image(tmp_path):
    from dataclasses import replace
    from turnitover.core.actions import QueryRuntime, RuntimeProperty
    calls = []
    def action(prompt, images):
        calls.append((prompt, images))
        return result({"action": {"type": "request_view", "view_id": "front"}})
    context = replace(CONTEXT, action_types=CONTEXT.action_types + ("query_runtime",),
                      runtime_properties=("stats",))
    history = [Observation(0, QueryRuntime(RuntimeProperty.STATS), None, {"triangles_scene": 321}, 2)]
    policy = ActionOnlyPolicy(action, None, tmp_path, "reference.png")
    policy.decide(context, history, 1)
    payload = json.loads(calls[0][0].rsplit("\n", 1)[1])
    assert payload["history"][0]["image_ref"] is None
    assert payload["history"][0]["payload"] == {"triangles_scene": 321}
    assert payload["image_order"] == ["reference.png"]
    assert calls[0][1] == [tmp_path / "reference.png"]
