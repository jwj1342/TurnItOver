from pathlib import Path

import pytest

from src.verifier import VerifierAction, VerifierEpisode


def test_verifier_episode_budget_and_finish(browser_session, fixture_dir: Path, tmp_path: Path) -> None:
    html = (fixture_dir / "valid_threejs.html").read_text(encoding="utf-8")
    episode = VerifierEpisode(browser_session, html, tmp_path / "episode", action_budget=2)

    initial = episode.start()
    assert initial.view_id == "view_00"
    assert initial.action == "initial"
    assert initial.remaining_budget == 2
    assert Path(initial.screenshot_path).is_file()

    first = episode.act(VerifierAction.ORBIT_RIGHT)
    second = episode.act(VerifierAction.ZOOM_IN)
    assert first.view_id == "view_01"
    assert second.view_id == "view_02"
    assert episode.remaining_budget == 0

    with pytest.raises(RuntimeError, match="budget exhausted"):
        episode.act(VerifierAction.CAPTURE)

    result = episode.finish(
        "The main component proportions differ from the reference.",
        selected_view_ids=[first.view_id, second.view_id],
    )
    assert result.feedback
    assert result.selected_view_ids == ["view_01", "view_02"]
    assert result.exhausted_budget is True
    assert (tmp_path / "episode" / "trace.json").is_file()


def test_verifier_finish_rejects_unknown_view(browser_session, fixture_dir: Path, tmp_path: Path) -> None:
    html = (fixture_dir / "valid_threejs.html").read_text(encoding="utf-8")
    episode = VerifierEpisode(browser_session, html, tmp_path / "episode", action_budget=1)
    episode.start()

    with pytest.raises(ValueError, match="unknown selected view ids"):
        episode.finish("Useful feedback", selected_view_ids=["view_99"])


def test_finish_requires_feedback(browser_session, fixture_dir: Path, tmp_path: Path) -> None:
    html = (fixture_dir / "valid_threejs.html").read_text(encoding="utf-8")
    episode = VerifierEpisode(browser_session, html, tmp_path / "episode", action_budget=1)
    episode.start()

    with pytest.raises(ValueError, match="non-empty feedback"):
        episode.finish("   ")
