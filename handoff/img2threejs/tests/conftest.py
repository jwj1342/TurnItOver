from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from src.runtime import BrowserConfig, BrowserSession


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def browser_session():
    executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    if not executable:
        executable = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")

    config = BrowserConfig(
        headless=True,
        settle_ms=80,
        canvas_timeout_ms=300,
        executable_path=executable,
    )

    try:
        session = BrowserSession(config)
        session.start()
    except Exception as exc:
        pytest.skip(f"Chromium unavailable for browser test: {exc}")

    try:
        yield session
    finally:
        session.close()
