"""Generator calls in the repair runner retry rate limits and server errors, and nothing else."""
import importlib.util
from pathlib import Path

import pytest

from turnitover.models.client import ModelError, ModelResult

SPEC = importlib.util.spec_from_file_location("run_repair_table", Path(__file__).parents[2] / "scripts/run_repair_table.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fake_complete(errors):
    calls = {"n": 0}
    def complete(config, prompt, images):
        calls["n"] += 1
        if errors:
            raise ModelError(errors.pop(0))
        return ModelResult("{}", "m", {}, "end_turn", 0.0)
    return complete, calls


def test_retries_rate_limit_then_succeeds(monkeypatch):
    complete, calls = fake_complete(["anthropic request failed with HTTP 429; no automatic retry",
                                     "anthropic request failed with HTTP 503; no automatic retry"])
    monkeypatch.setattr(runner, "complete", complete)
    monkeypatch.setattr(runner, "RETRY_WAITS", (0, 0, 0))
    assert runner.complete_with_retry(None, "p", []).text == "{}"
    assert calls["n"] == 3


def test_does_not_retry_client_errors(monkeypatch):
    complete, calls = fake_complete(["anthropic request failed with HTTP 400; no automatic retry"])
    monkeypatch.setattr(runner, "complete", complete)
    monkeypatch.setattr(runner, "RETRY_WAITS", (0, 0, 0))
    with pytest.raises(ModelError):
        runner.complete_with_retry(None, "p", [])
    assert calls["n"] == 1


def test_gives_up_after_bounded_retries(monkeypatch):
    complete, calls = fake_complete(["x HTTP 429 y"] * 10)
    monkeypatch.setattr(runner, "complete", complete)
    monkeypatch.setattr(runner, "RETRY_WAITS", (0, 0))
    with pytest.raises(ModelError):
        runner.complete_with_retry(None, "p", [])
    assert calls["n"] == 3
