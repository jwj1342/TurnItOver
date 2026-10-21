import hashlib
import json

import pytest

from turnitover.models.client import ModelResult
from turnitover.verifier.contracts import Context
from turnitover.verifier.loop import run_verification
from turnitover.verifier.policies import VisionPolicy
from turnitover.verifier.runner import VerifyConfig


CONTEXT = Context("Inspect runtime", (), (), ("front",), ("stats",), ("query_runtime",), 5000, 32)
ACTION = {"action": {"type": "query_runtime", "property": "stats"}}
VERDICT = {"verdict": {"status": "uncertain", "confidence": 0.5, "findings": [],
                       "summary": "Runtime inspected", "limitations": ["Only runtime was observed"]}}


class RuntimeSession:
    def __init__(self):
        self.queries = []

    def query_runtime(self, prop):
        self.queries.append(prop)
        return {"triangles_scene": 12, "draw_calls": 1}


def model_result(value):
    return ModelResult(json.dumps(value), "frozen-model", {}, "completed", 1.0)


def test_exact_active_rejects_early_stop_and_matches_fixed_final_prompt(tmp_path):
    prompts = {"active": [], "fixed": []}
    episodes = {}
    sessions = {}
    for mode in prompts:
        output = tmp_path / mode
        output.mkdir()

        def call(prompt, images, mode=mode):
            prompts[mode].append(prompt)
            if mode == "fixed":
                # The complete fixed sequence must be committed before final inference.
                assert (output / "observation_plan.json").is_file()
            elif len(prompts[mode]) == 1:
                return model_result(VERDICT)  # Invalid early stop must trigger schema feedback.
            data = json.loads(prompt.split("Observed task data:\n")[1].split("\n")[0])
            return model_result(ACTION if data["remaining"] else VERDICT)

        policy = VisionPolicy(call, output, [], mode, exact_budget=True)
        session = RuntimeSession()
        episodes[mode] = run_verification(session, policy, CONTEXT, 3, lambda *_: "unused")
        sessions[mode] = session

    assert all(episode.spent == 3 and len(episode.trajectory) == 3 for episode in episodes.values())
    assert all(episode.termination == "budget_exhausted" for episode in episodes.values())
    assert all(len(session.queries) == 3 for session in sessions.values())
    assert len(prompts["active"]) == 5 and len(prompts["fixed"]) == 1
    assert "Exact observation budget requires an action" in prompts["active"][1]
    active_protocol, active_data = prompts["active"][-1].split("Observed task data:\n")
    fixed_protocol, fixed_data = prompts["fixed"][-1].split("Observed task data:\n")
    assert active_protocol == fixed_protocol
    active_data, fixed_data = json.loads(active_data), json.loads(fixed_data)
    for data in (active_data, fixed_data):
        for observation in data["history"]:
            observation.pop("elapsed_ms")  # Real execution latency is recorded separately per arm.
    assert active_data == fixed_data

    plan = json.loads((tmp_path / "fixed/observation_plan.json").read_text())
    assert plan["base_actions"] == [ACTION["action"]]
    assert plan["planned_actions"] == [ACTION["action"]] * 3
    serialized = json.dumps(plan["planned_actions"], sort_keys=True, separators=(",", ":"))
    assert plan["sha256"] == hashlib.sha256(serialized.encode()).hexdigest()


def test_exact_active_repeated_early_verdict_fails_without_observing(tmp_path):
    policy = VisionPolicy(lambda *_: model_result(VERDICT), tmp_path, [], exact_budget=True)
    session = RuntimeSession()
    episode = run_verification(session, policy, CONTEXT, 2, lambda *_: "unused")
    assert episode.termination == "invalid_decision" and episode.spent == 0
    assert len(policy.calls) == 3 and not session.queries


def test_default_active_keeps_early_stop_compatibility(tmp_path):
    policy = VisionPolicy(lambda *_: model_result(VERDICT), tmp_path, [])
    episode = run_verification(RuntimeSession(), policy, CONTEXT, 3, lambda *_: "unused")
    assert episode.termination == "judge_finished" and episode.spent == 0
    assert len(policy.calls) == 1


@pytest.mark.parametrize("config", [VerifyConfig(exact_budget=1),
                                  VerifyConfig(exact_budget=True, mode="runtime"),
                                  VerifyConfig(exact_budget=True, mode="random")])
def test_exact_budget_configuration_rejects_unsupported_values(config):
    with pytest.raises(ValueError):
        config.validate()


def test_exact_fixed_with_no_legal_action_fails_explicitly(tmp_path):
    context = Context("Inspect", (), (), ("front",), (), (), 5000, 32)
    policy = VisionPolicy(lambda *_: pytest.fail("Cannot infer without an available action"),
                          tmp_path, [], "fixed", exact_budget=True)
    episode = run_verification(RuntimeSession(), policy, context, 1, lambda *_: "unused")
    assert episode.termination == "invalid_decision" and episode.spent == 0
    assert "no available observation actions" in episode.error_message
