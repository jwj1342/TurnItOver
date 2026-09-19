from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

from PIL import Image

from src.agents import CodeAgent, VerifierAgent
from src.config import Settings
from src.pipeline import PipelineController
from src.verifier import RelativeDiscreteActionSpace
from src.agents import CodeAgentResult, VerifierAgentResult
from src.runtime.render_check import RenderCheckResult


def _settings() -> Settings:
    return Settings(
        qwen_base_url="https://example.test/v1",
        qwen_api_key="test",
        qwen_model="openai/qwen-test",
        max_render_retries=2,
        max_visual_revisions=2,
        verifier_action_budget=2,
    )


@pytest.mark.parametrize("success_at", [0, 3, None])
@pytest.mark.parametrize("fail_after_visual_revision", [False, True])
def test_render_gate_and_three_repairs(
    tmp_path, monkeypatch, success_at, fail_after_visual_revision
):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (8, 8)).save(reference)
    checks = []
    repairs = []
    verified = []
    revision = [False]

    def code_result(workspace, html):
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / "candidate.html"
        path.write_text(html)
        return CodeAgentResult(html=html)

    def generate(ref, workspace):
        return code_result(workspace, "initial")

    def revise(ref, html, feedback, evidence, workspace):
        revision[0] = True
        return code_result(workspace, "revised")

    def repair(ref, html, previous, workspace):
        assert previous.success is False
        repairs.append(html)
        return code_result(workspace, f"repair-{len(repairs)}")

    def check(session, load, directory):
        checks.append(load)
        bypass = fail_after_visual_revision and not revision[0]
        ok = bypass or (success_at is not None and len(repairs) == success_at)
        return RenderCheckResult(
            success=ok, page_loaded=True, canvas_found=ok, canvas_nonempty=ok
        )

    def verify(ref, html, directory):
        verified.append(html)
        verdict = "revise" if fail_after_visual_revision and not revision[0] else "accept"
        return VerifierAgentResult(verdict=verdict, feedback="check structure", selected_view_ids=[])

    monkeypatch.setattr("src.pipeline.retry.check_render", check)
    controller = PipelineController(
        settings=replace(_settings(), max_render_retries=3),
        code_agent=SimpleNamespace(generate=generate, repair_runtime=repair, revise_visual=revise),
        verifier_agent=None,
        browser_session=SimpleNamespace(load_html=lambda html: html),
        action_space=RelativeDiscreteActionSpace(),
    )
    monkeypatch.setattr(controller, "_run_fresh_verifier", verify)
    run_dir = tmp_path / "run"
    result = controller.run(reference, run_dir)
    expected_repairs = 3 if success_at is None else success_at
    assert len(repairs) == expected_repairs
    assert len(checks) == 1 + expected_repairs + int(fail_after_visual_revision)
    assert len(verified) == int(fail_after_visual_revision) + int(success_at is not None)
    assert result.status == ("render_failed" if success_at is None else "accepted")
    assert result.rounds[-1].render_attempts == 1 + expected_repairs
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == result.status
    assert (run_dir / "final.html").read_text() == result.final_html
    reports = list(run_dir.glob("round_*/render_attempts/attempt_*/check.json"))
    assert len(reports) == len(checks)


def test_pipeline_revises_then_reverifies_fresh(
    browser_session,
    fixture_dir: Path,
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), color="white").save(reference)
    html = (fixture_dir / "valid_threejs.html").read_text(encoding="utf-8")

    def generate_runner(workspace: Path, prompt: str, reference_image: Path):
        assert reference_image == reference
        (workspace / "candidate.html").write_text(html, encoding="utf-8")
        return {"reasoning": "initial generation"}

    visual_calls = []

    def visual_runner(workspace: Path, prompt: str, reference_image: Path, evidence_images: list[Path]):
        visual_calls.append(list(evidence_images))
        assert "main component" in prompt
        assert evidence_images
        assert all(path.is_file() for path in evidence_images)
        (workspace / "candidate.html").write_text(html, encoding="utf-8")
        return {"reasoning": "localized visual revision"}

    code_agent = CodeAgent(
        _settings(),
        runner=generate_runner,
        visual_runner=visual_runner,
    )

    verifier_rounds = []

    def verifier_runner(episode, reference_image: Path, prompt: str):
        verifier_rounds.append(Path(episode.output_dir))
        if len(verifier_rounds) == 1:
            evidence = episode.act("capture")
            episode.finish(
                "The main component should be wider.",
                selected_view_ids=[evidence.view_id],
                verdict="revise",
            )
        else:
            episode.finish(
                "No important actionable mismatch remains.",
                selected_view_ids=[episode.observations[0].view_id],
                verdict="accept",
            )
        return {"reasoning": f"verifier round {len(verifier_rounds)}"}

    verifier_agent = VerifierAgent(_settings(), runner=verifier_runner)
    original_load_html = browser_session.load_html
    loaded_html = []

    def load_html_once_per_round(candidate: str):
        loaded_html.append(candidate)
        return original_load_html(candidate)

    browser_session.load_html = load_html_once_per_round
    run_dir = tmp_path / "run"
    controller = PipelineController(
        settings=_settings(),
        code_agent=code_agent,
        verifier_agent=verifier_agent,
        browser_session=browser_session,
        action_space=RelativeDiscreteActionSpace(),
    )

    result = controller.run(reference, run_dir)

    assert result.status == "accepted"
    assert result.accepted is True
    assert result.visual_revisions == 1
    assert len(result.rounds) == 2
    assert [item.verifier_verdict for item in result.rounds] == ["revise", "accept"]

    assert len(verifier_rounds) == 2
    assert loaded_html == [html, html]
    assert verifier_rounds[0] != verifier_rounds[1]
    assert "round_00" in str(verifier_rounds[0])
    assert "round_01" in str(verifier_rounds[1])

    assert len(visual_calls) == 1
    assert "round_00" in str(visual_calls[0][0])
    assert (run_dir / "round_00" / "verifier" / "trace.json").is_file()
    assert (run_dir / "round_01" / "verifier" / "trace.json").is_file()
    assert (run_dir / "trajectory.json").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "final.html").is_file()
