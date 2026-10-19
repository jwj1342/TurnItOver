import json
from dataclasses import asdict

import pytest

from turnitover.core.actions import Observation, RequestView
from turnitover.models.client import ModelResult
from turnitover.oracle.supervision import action_examples
from turnitover.verifier.action_only import ACTION_STAGE_INSTRUCTION, ActionOnlyPolicy, complete_final_readout
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
    assert calls[0][0].replace("\n" + ACTION_STAGE_INSTRUCTION, "", 1) == exported["messages"][0]["content"]
    assert calls[0][0].count(ACTION_STAGE_INSTRUCTION) == 1
    protocol, payload = calls[0][0].rsplit("\n", 1)
    assert protocol.endswith(ACTION_STAGE_INSTRUCTION)
    assert "Do not return a verdict or stop early" in protocol
    assert json.loads(payload)["remaining"] == 3
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


def test_inference_failure_is_preserved_without_retry(tmp_path):
    from turnitover.models.client import ModelError
    def failed(prompt, images):
        raise ModelError("SSH bridge unavailable")
    policy = ActionOnlyPolicy(failed, None, tmp_path, "reference.png")
    with pytest.raises(ModelError, match="SSH bridge unavailable"):
        policy.decide(CONTEXT, (), 2)
    saved = json.loads((tmp_path / "action_calls/000/metadata.json").read_text())
    assert saved["error_type"] == "ModelError"
    assert policy.action_attempts == 1 and policy.legal_actions == 0


def test_action_failure_still_gets_frozen_readout_on_actual_prefix(tmp_path):
    from turnitover.verifier.loop import Episode
    from turnitover.verifier.contracts import uncertain
    calls = []
    verdict = {"verdict": {"status": "pass", "confidence": .7, "findings": [],
                          "summary": "No defect observed", "limitations": ["Limited prefix"]}}
    def final(prompt, images):
        calls.append(prompt)
        return result(verdict)
    policy = ActionOnlyPolicy(None, final, tmp_path, "reference.png")
    history = (Observation(0, RequestView("front"), "first.png", {}, 20),)
    failed = Episode(uncertain("Action invalid"), "invalid_decision", history, 1,
                     "DecisionError", error_message="Unknown view")
    readout, meta = complete_final_readout(policy, CONTEXT, failed)
    assert readout.verdict.status == "pass"
    assert readout.termination == "invalid_decision" and readout.spent == 1
    assert readout.trajectory == history and readout.error_message == "Unknown view"
    assert '"remaining": 0' in calls[0] and '"step": 0' in calls[0]
    assert meta == dict(supplemental=True, calls=1, status="complete")
    again, meta = complete_final_readout(policy, CONTEXT, readout)
    assert again == readout and len(calls) == 1 and not meta["supplemental"]


def test_supplemental_final_failure_is_separate_from_action_failure(tmp_path):
    from turnitover.models.client import ModelError
    from turnitover.verifier.loop import Episode
    from turnitover.verifier.contracts import uncertain
    def final(prompt, images):
        raise ModelError("Final transport failure")
    policy = ActionOnlyPolicy(None, final, tmp_path, "reference.png")
    failed = Episode(uncertain("Action invalid"), "invalid_decision", (), 0, "DecisionError")
    readout, meta = complete_final_readout(policy, CONTEXT, failed)
    assert readout == failed
    assert meta["error_type"] == "ModelError" and meta["calls"] == 1
    assert meta["error"] == "Final transport failure"


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
