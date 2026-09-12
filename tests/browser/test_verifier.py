import json

import pytest

from turnitover.core.program import ObjectProgram
from turnitover.models.client import ModelResult
from turnitover.verifier.runner import VerifyConfig, verify

pytestmark = pytest.mark.browser


def test_runtime_verifier_detects_budget_violation(tmp_path, toy_spec, render_config, views):
    bad = toy_spec.replace_part("drawer_0", segments=(16, 16, 16)).replace_part("drawer_1", segments=(16, 16, 16))
    out = tmp_path / "verify"
    result = verify(ObjectProgram.from_spec(bad), out, render_config, views, VerifyConfig(budget=2, mode="runtime"),
                    reference_program=ObjectProgram.from_spec(toy_spec))
    assert result["verdict"]["status"] == "fail" and result["spent"] == 2
    assert result["verdict"]["findings"][0]["evidence_steps"] == (0,)
    assert result["audit"]["status"] == "complete" and not result["audit"]["used_by_judge"]
    assert (out / "observations/001.png").exists() and (out / "index.html").exists()
    assert json.loads((out / "feedback.json").read_text())["verdict"]["status"] == "fail"


def test_clean_runtime_is_uncertain_and_program_error_is_not_pass(tmp_path, toy_program, render_config, views):
    good = verify(toy_program, tmp_path / "good", render_config, views, VerifyConfig(budget=1, mode="runtime"))
    assert good["verdict"]["status"] == "uncertain"
    bad = verify(ObjectProgram("export default 42"), tmp_path / "bad", render_config, views, VerifyConfig(mode="runtime"))
    assert bad["status"] == "error" and bad["termination"] == "program_error"
    assert bad["verdict"]["status"] != "pass"


def test_active_model_browser_roundtrip(tmp_path, toy_program, render_config, views):
    calls = []
    def judge(prompt, images):
        calls.append(images)
        assert all(image.is_file() for image in images)
        decision = {"action": {"type": "request_view", "view_id": "front"}} if len(calls) == 1 else {
            "verdict": {"status": "uncertain", "confidence": 0.6, "findings": [], "summary": "Need reference",
                        "limitations": ["No reference image"]}}
        return ModelResult(json.dumps(decision), "mock-vision", {}, "completed", 1)
    result = verify(toy_program, tmp_path / "active", render_config, views, VerifyConfig(budget=1), model_call=judge)
    assert len(calls) == 2 and len(calls[1]) == 1
    assert result["termination"] == "budget_exhausted" and result["spent"] == 1
