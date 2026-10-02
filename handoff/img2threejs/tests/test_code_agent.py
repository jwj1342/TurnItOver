from pathlib import Path

from PIL import Image

from src.agents import CodeAgent
from src.config import Settings
from src.runtime.render_check import RenderCheckResult


VALID_HTML = """<!doctype html><html><body><canvas id='c'></canvas><script>document.getElementById('c').width=64;document.getElementById('c').height=64;const ctx=document.getElementById('c').getContext('2d');ctx.fillStyle='red';ctx.fillRect(0,0,64,64);</script></body></html>"""


def _settings() -> Settings:
    return Settings(
        qwen_base_url="https://example.test/v1",
        qwen_api_key="test",
        qwen_model="openai/qwen-test",
    )


def _reference(tmp_path: Path) -> Path:
    path = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), color="white").save(path)
    return path


def test_generate_reads_candidate_from_workspace(tmp_path: Path) -> None:
    def fake_runner(workspace: Path, prompt: str, reference_image: Path):
        assert "candidate.html" in prompt
        assert "RoomEnvironment" in prompt
        assert "不得只使用强 AmbientLight" in prompt
        assert "toneMappingExposure = 1.0" in prompt
        assert "位置 `(5, 8, 6)`" in prompt
        assert reference_image.is_file()
        (workspace / "candidate.html").write_text(VALID_HTML, encoding="utf-8")
        return {
            "reasoning": "Build a simple visible object first.",
            "final_answer": "Done",
            "events": [{"type": "fake"}],
            "usage": {"tokens": 123},
        }

    agent = CodeAgent(_settings(), runner=fake_runner)
    result = agent.generate(_reference(tmp_path), tmp_path / "workspace")

    assert result.html == VALID_HTML
    assert result.reasoning == "Build a simple visible object first."
    assert result.events == [{"type": "fake"}]


def test_runtime_repair_receives_structured_failure(tmp_path: Path) -> None:
    captured = {}

    def fake_runner(workspace: Path, prompt: str, reference_image: Path):
        captured["prompt"] = prompt
        assert "fixed neutral evaluation environment" in prompt
        assert (workspace / "candidate.html").read_text(encoding="utf-8") == "<html>broken</html>"
        (workspace / "candidate.html").write_text(VALID_HTML, encoding="utf-8")
        return {"reasoning": "Fix the missing render path."}

    failure = RenderCheckResult(
        success=False,
        page_loaded=True,
        canvas_found=False,
        canvas_nonempty=False,
        page_errors=["ReferenceError: THREE is not defined"],
    )
    agent = CodeAgent(_settings(), runner=fake_runner)
    result = agent.repair_runtime(
        _reference(tmp_path),
        "<html>broken</html>",
        failure,
        tmp_path / "repair-workspace",
    )

    assert "THREE is not defined" in captured["prompt"]
    assert result.html == VALID_HTML


def test_visual_revision_receives_feedback_and_evidence(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    evidence = tmp_path / "view_01.png"
    Image.new("RGB", (32, 32), color="gray").save(evidence)
    captured = {}

    def fake_visual_runner(
        workspace: Path,
        prompt: str,
        reference_image: Path,
        evidence_images: list[Path],
    ):
        captured["prompt"] = prompt
        captured["evidence"] = evidence_images
        assert reference_image == reference
        assert (workspace / "candidate.html").read_text(encoding="utf-8") == VALID_HTML
        (workspace / "candidate.html").write_text(VALID_HTML.replace("red", "blue"), encoding="utf-8")
        return {"reasoning": "Widen the main component based on the selected side view."}

    agent = CodeAgent(_settings(), visual_runner=fake_visual_runner)
    result = agent.revise_visual(
        reference,
        VALID_HTML,
        "The main component should be wider.",
        [evidence],
        tmp_path / "visual-workspace",
    )

    assert "main component should be wider" in captured["prompt"]
    assert "RoomEnvironment" in captured["prompt"]
    assert "不要在缺少诊断时只反复修改颜色值" in captured["prompt"]
    assert captured["evidence"] == [evidence]
    assert "blue" in result.html
