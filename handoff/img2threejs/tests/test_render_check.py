from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.retry import run_with_render_retry
from src.runtime.errors import classify_render_failure
from src.runtime.render_check import check_render


pytestmark = pytest.mark.browser


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_valid_canvas_passes(browser_session, fixture_dir: Path, tmp_path: Path) -> None:
    load = browser_session.load_html(_read(fixture_dir / "valid_threejs.html"))
    result = check_render(browser_session, load, tmp_path / "valid")

    assert result.success is True
    assert result.page_loaded is True
    assert result.canvas_found is True
    assert result.canvas_nonempty is True
    assert result.screenshot_path is not None
    assert Path(result.screenshot_path).is_file()


def test_blank_canvas_fails(browser_session, fixture_dir: Path, tmp_path: Path) -> None:
    load = browser_session.load_html(_read(fixture_dir / "blank_canvas.html"))
    result = check_render(browser_session, load, tmp_path / "blank")

    assert result.success is False
    assert result.canvas_found is True
    assert result.canvas_nonempty is False
    failure = classify_render_failure(**{
        "page_loaded": result.page_loaded,
        "canvas_found": result.canvas_found,
        "canvas_nonempty": result.canvas_nonempty,
        "load_errors": result.load_errors,
        "page_errors": result.page_errors,
        "console_errors": result.console_errors,
    })
    assert failure is not None
    assert failure.kind == "blank_canvas"


def test_runtime_error_fails(browser_session, fixture_dir: Path, tmp_path: Path) -> None:
    load = browser_session.load_html(_read(fixture_dir / "runtime_error.html"))
    result = check_render(browser_session, load, tmp_path / "runtime-error")

    assert result.success is False
    assert result.page_errors
    assert "intentional runtime failure" in result.repair_message()


def test_syntax_error_fails(browser_session, fixture_dir: Path, tmp_path: Path) -> None:
    load = browser_session.load_html(_read(fixture_dir / "syntax_error.html"))
    result = check_render(browser_session, load, tmp_path / "syntax-error")

    assert result.success is False
    assert result.page_errors or result.console_errors


def test_retry_replaces_failed_candidate(
    browser_session,
    fixture_dir: Path,
    tmp_path: Path,
) -> None:
    blank = _read(fixture_dir / "blank_canvas.html")
    valid = _read(fixture_dir / "valid_threejs.html")
    calls: list[tuple[int, bool]] = []

    def generate_or_repair(previous, attempt_index: int) -> str:
        calls.append((attempt_index, previous is not None))
        return blank if attempt_index == 0 else valid

    result = run_with_render_retry(
        browser_session,
        generate_or_repair,
        tmp_path / "retry",
        max_attempts=3,
    )

    assert result.success is True
    assert len(result.attempts) == 2
    assert result.attempts[0].check.success is False
    assert result.attempts[1].check.success is True
    assert calls == [(0, False), (1, True)]
