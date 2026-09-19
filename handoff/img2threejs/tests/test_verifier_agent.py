from pathlib import Path

import json

import pytest

from src.agents import VerifierAgent
from src.agents.verifier_agent import _run_with_finish_retry
from src.config import Settings
from src.verifier import VerifierAction, VerifierEpisode


def _settings() -> Settings:
    return Settings(
        qwen_base_url="https://example.test/v1",
        qwen_api_key="secret",
        qwen_model="openai/qwen-test",
        verifier_action_budget=3,
    )


def test_verifier_agent_fake_runner_active_episode(
    browser_session,
    fixture_dir: Path,
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    # The fake runner never decodes the reference, but VerifierAgent requires a real file.
    reference.write_bytes(b"not-an-image-for-fake-runner")
    html = (fixture_dir / "valid_threejs.html").read_text(encoding="utf-8")
    episode = VerifierEpisode(browser_session, html, tmp_path / "verify", action_budget=3)

    def fake_runner(active_episode, reference_image, prompt):
        assert reference_image == reference
        assert "finish" in prompt
        assert "先结构、后外观" in prompt
        assert "仅凭截图不得直接断言是材质参数错误" in prompt
        first = active_episode.act(VerifierAction.ORBIT_RIGHT)
        second = active_episode.act(VerifierAction.CAPTURE)
        active_episode.finish(
            "The dominant component should be wider and the secondary part should sit closer to it.",
            selected_view_ids=[first.view_id, second.view_id],
        )
        return {
            "reasoning": "I need another side view before judging proportions.",
            "final_answer": "finished",
            "events": [{"type": "fake"}],
            "usage": {"tokens": 1},
        }

    result = VerifierAgent(_settings(), runner=fake_runner).verify(reference, episode)
    assert result.feedback.startswith("The dominant component")
    assert result.selected_view_ids == ["view_01", "view_02"]
    assert len(result.observations) == 3
    assert result.reasoning is not None
    assert (tmp_path / "verify" / "trace.json").is_file()


def test_finish_retry_runs_at_most_once() -> None:
    class Episode:
        finished = False

    class Conversation:
        def __init__(self) -> None:
            self.runs = 0
            self.messages = []

        def run(self) -> None:
            self.runs += 1

        def send_message(self, message) -> None:
            self.messages.append(message)

    conversation = Conversation()
    attempted = _run_with_finish_retry(conversation, Episode(), "finish now")

    assert attempted is True
    assert conversation.runs == 2
    assert conversation.messages == ["finish now"]


def test_incomplete_verifier_writes_diagnostic(
    browser_session,
    fixture_dir: Path,
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"not-an-image-for-fake-runner")
    html = (fixture_dir / "valid_threejs.html").read_text(encoding="utf-8")
    output_dir = tmp_path / "verify"
    episode = VerifierEpisode(browser_session, html, output_dir, action_budget=3)

    def incomplete_runner(active_episode, reference_image, prompt):  # noqa: ARG001
        active_episode.act(VerifierAction.ORBIT_LEFT)
        return {
            "reasoning": "I inspected the side but forgot to finish.",
            "final_answer": "The shape needs work.",
            "events": [{"type": "fake-incomplete"}],
            "usage": {"tokens": 7},
            "finish_retry_attempted": True,
        }

    with pytest.raises(RuntimeError, match="after one finish-only retry"):
        VerifierAgent(_settings(), runner=incomplete_runner).verify(reference, episode)

    payload = json.loads((output_dir / "incomplete_agent.json").read_text(encoding="utf-8"))
    assert payload["finish_retry_attempted"] is True
    assert payload["remaining_budget"] == 2
    assert len(payload["observations"]) == 2
    assert payload["events"] == [{"type": "fake-incomplete"}]
