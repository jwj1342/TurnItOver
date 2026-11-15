"""Generator calls in the repair runner: budgeted retry of transient errors only, caching of finished responses only,
and the scripted fix working on the elided program."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from turnitover.models.client import ModelError, ModelResult, TransientModelError
from turnitover.repair.patches import apply_response, elide, expand

SPEC = importlib.util.spec_from_file_location("run_repair_table", Path(__file__).parents[2] / "scripts/run_repair_table.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fake_world(monkeypatch, errors, finish="end_turn"):
    """Replace the API call and the clock; errors are raised in order, then the call succeeds."""
    clock = SimpleNamespace(now=0.0, waits=[], calls=0)
    def complete(config, prompt, images):
        clock.calls += 1
        if errors:
            raise errors.pop(0)
        return ModelResult('{"stop": true}', "m", {}, finish, 0.0)
    def sleep(seconds):
        clock.waits.append(seconds)
        clock.now += seconds
    monkeypatch.setattr(runner, "complete", complete)
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    return clock


def test_retries_transient_errors_with_doubling_backoff_and_retry_after(monkeypatch, capsys):
    clock = fake_world(monkeypatch, [TransientModelError("anthropic", "HTTP 429", retry_after=45.0),
                                     TransientModelError("anthropic", "HTTP 503"),
                                     TransientModelError("openai_compatible", "URLError/ConnectionRefusedError")])
    assert runner.complete_with_retry(None, "p", []).text == '{"stop": true}'
    assert clock.calls == 4 and clock.waits == [45.0, 40, 80]
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [l["generator_retry"] for l in lines] == ["HTTP 429", "HTTP 503", "URLError/ConnectionRefusedError"]
    assert [l["elapsed_s"] for l in lines] == [0.0, 45.0, 85.0]


def test_client_errors_raise_immediately(monkeypatch):
    clock = fake_world(monkeypatch, [ModelError("anthropic request failed with HTTP 400")])
    with pytest.raises(ModelError, match="HTTP 400"):
        runner.complete_with_retry(None, "p", [])
    assert clock.calls == 1 and clock.waits == []


def test_waits_are_capped_and_the_budget_ends_retries(monkeypatch):
    monkeypatch.setenv("TIO_GENERATOR_RETRY_BUDGET_S", "1200")
    clock = fake_world(monkeypatch, [TransientModelError("x", "HTTP 503", retry_after=900.0)]
                       + [TransientModelError("x", "HTTP 502") for _ in range(20)])
    with pytest.raises(TransientModelError):
        runner.complete_with_retry(None, "p", [])
    assert clock.waits == [300, 40, 80, 160, 300, 300] and sum(clock.waits) + 300 > 1200


def test_truncated_responses_are_not_cached(monkeypatch, tmp_path):
    clock = fake_world(monkeypatch, [], finish="length")
    generator = runner.cached_api_generator(None, "g", tmp_path)
    assert generator("p", []).finish_reason == "length"
    assert list((tmp_path / "generator" / "g").iterdir()) == []
    monkeypatch.setattr(runner, "complete", lambda *a: ModelResult('{"stop": true}', "m", {}, "stop", 0.0))
    generator("p", [])
    monkeypatch.setattr(runner, "complete", lambda *a: pytest.fail("cached response not used"))
    assert generator("p", []).finish_reason == "stop" and clock.calls == 1


def test_scripted_fix_edits_only_text_between_geometry_placeholders():
    geometry = "[" + ", ".join(f"{i * 0.5:.1f}" for i in range(100)) + "]"
    initial = f"const g = {geometry};\nconst axis = [1, 0, 0];\nconst h = {geometry};\n"
    reference = initial.replace("[1, 0, 0]", "[0, 0, 1]")
    case = SimpleNamespace(initial=initial, reference=SimpleNamespace(source=reference))
    response = json.loads(runner.scripted_generator("scripted-fix", case)("prompt", []).text)
    shown, arrays = elide(initial)
    assert expand(apply_response(shown, response), arrays) == reference


def test_generator_error_episodes_are_rerun(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"status": "generator_error", "error": "HTTP 503"}))
    assert not runner.episode_done(tmp_path)
