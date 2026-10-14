import json

import pytest
from PIL import Image

from turnitover.teacher.audit import audit_episode
from turnitover.teacher.episode import run_teacher_episode
from turnitover.teacher.policy import TeacherPolicy
from turnitover.teacher.providers import MockTeacher
from turnitover.verifier.contracts import Context


pytestmark = pytest.mark.browser


def test_teacher_real_browser_observation_is_returned_in_causal_order(
    tmp_path, session_factory, toy_program, views
):
    info = session_factory.load(toy_program, framing=None)
    reference = tmp_path / "reference-00.png"
    Image.new("RGB", (8, 8), "blue").save(reference)
    decisions = [
        {"action": {"type": "actuate_joint", "joint_id": "door", "detent": "limit"}},
        {"verdict": {"status": "uncertain", "confidence": 0.0, "findings": [],
                     "summary": "bounded fixture", "limitations": ["one observation"]}},
    ]
    context = Context("inspect", info.parts, tuple(info.joints), tuple(view.id for view in views),
                      ("joint_state",), ("request_view", "actuate_joint", "query_runtime"), 5000, 32)
    policy = TeacherPolicy(MockTeacher(decisions), tmp_path, [reference])

    episode = run_teacher_episode(session_factory, policy, context, 1, tmp_path, episode_id="browser")
    audit = audit_episode(tmp_path)
    second_request = json.loads((tmp_path / "model_calls/001/request.json").read_text())

    assert episode["termination"] == "budget_exhausted" and audit["pipeline_ok"]
    assert episode["budget_exhausted"] and not episode["teacher_verdict_missing"]
    assert (tmp_path / "observations/000.png").is_file()
    assert second_request["remaining"] == 0 and second_request["history"][0]["step"] == 0
    assert second_request["stop_required"] is True
    assert second_request["image_order"][-1]["kind"] == "observation"
