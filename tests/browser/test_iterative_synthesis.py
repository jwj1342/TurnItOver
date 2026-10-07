import json

import pytest
from PIL import Image

from turnitover.models.client import ModelResult
from turnitover.repair.iterative import IterativeConfig, run_iterative
from turnitover.verifier.runner import VerifyConfig


pytestmark = pytest.mark.browser


def test_iterative_pipeline_uses_real_render_gate_and_browser_verifier(
    tmp_path, toy_program, render_config, views
):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), "blue").save(reference)
    judge_calls = []

    def generator(prompt, images):
        assert prompt == "build toy cabinet"
        assert len(images) == 1 and images[0].is_file()
        return ModelResult(f"```ts\n{toy_program.source}\n```", "fixture-generator", {"total_tokens": 3},
                           "completed", 1.0)

    def judge(prompt, images):
        judge_calls.append(images)
        assert all(image.is_file() for image in images)
        decision = {"action": {"type": "request_view", "view_id": "front"}} if len(judge_calls) == 1 else {
            "verdict": {"status": "pass", "confidence": 1.0, "findings": [],
                        "summary": "Fixture verified after one browser observation.", "limitations": []}
        }
        return ModelResult(json.dumps(decision), "fixture-judge", {"total_tokens": 2}, "completed", 1.0)

    output = tmp_path / "iterative"
    result = run_iterative(
        (reference,), output, render_config, views,
        IterativeConfig(max_runtime_repairs=0, max_visual_revisions=0, verifier=VerifyConfig(budget=1)),
        generator, generation_prompt="build toy cabinet", judge_call=judge,
    )

    assert result["status"] == "accepted" and result["accepted"]
    assert result["model_calls"] == {"generation": 1, "runtime_repair": 0, "visual_revision": 0, "judge": 2}
    assert result["rounds"][0]["program_sha"] == toy_program.sha
    assert (output / "round-000/render-gate/attempt-000/render.png").is_file()

    gate = json.loads((output / "round-000/render-gate/attempt-000/result.json").read_text())
    verifier = json.loads((output / "round-000/verifier/result.json").read_text())
    trajectory = [json.loads(line) for line in (output / "round-000/verifier/trajectory.jsonl").read_text().splitlines()]

    assert gate["success"] and gate["image_ref"] == "render.png"
    assert verifier["status"] == "complete" and verifier["verdict"]["status"] == "pass"
    assert trajectory[0]["action"] == {"view_id": "front", "__type__": "RequestView"}
    assert len(judge_calls) == 2 and len(judge_calls[1]) == 2
