"""Offline checks of wire records, shared limits, and model identity gates."""
import io
import json
from urllib.error import HTTPError

import pytest
from PIL import Image

from turnitover.api_pipeline.api import FrozenAPI
from turnitover.models.client import ModelError, complete
from turnitover.models.config import ModelConfig


def config(provider="openai_compatible"):
    return ModelConfig("teacher", provider, "k3-256k", "https://private.example/v1", "secret-key")


def response(model="served-v1", cost=None):
    usage = {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20}
    if cost is not None:
        usage["cost"] = cost
    return {"model": model, "usage": usage,
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]}


def transport(monkeypatch, replies):
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(json.loads(request.data))
            item = next(replies)
            if isinstance(item, Exception):
                raise item
            return io.BytesIO(json.dumps(item).encode())

    monkeypatch.setattr("turnitover.models.client.build_opener", lambda *a: Opener())
    return requests


def load(path):
    return json.loads(path.read_text())


def test_trace_contains_full_image_payload_without_credentials(tmp_path, monkeypatch):
    photo = tmp_path / "input.png"
    Image.new("RGB", (4, 4), "blue").save(photo)
    raw = response(cost=0.012)
    sent = transport(monkeypatch, iter([raw]))
    trace = tmp_path / "trace"
    result = complete(config(), "inspect", [photo], trace_dir=trace, temperature=0)
    assert result.text == "answer"
    assert load(trace / "request.json") == sent[0]
    assert sent[0]["temperature"] == 0
    assert "data:image/png;base64," in json.dumps(sent[0])
    assert load(trace / "response.json") == raw
    assert load(trace / "metadata.json")["provider_cost"] == 0.012
    archive = "".join(p.read_text() for p in trace.iterdir())
    assert "secret-key" not in archive and "private.example" not in archive


def test_default_request_has_no_temperature_and_no_trace(tmp_path, monkeypatch):
    sent = transport(monkeypatch, iter([response()]))
    complete(config(), "inspect")
    assert "temperature" not in sent[0]
    assert not list(tmp_path.iterdir())


def test_gemini_temperature_nested_in_generation_config(tmp_path, monkeypatch):
    raw = {"modelVersion": "served-v1", "candidates": [
        {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}]}
    sent = transport(monkeypatch, iter([raw]))
    complete(config("gemini"), "inspect", trace_dir=tmp_path, temperature=0)
    assert sent[0]["generationConfig"]["temperature"] == 0
    assert "temperature" not in {k: v for k, v in sent[0].items() if k != "generationConfig"}


def test_http_failure_records_only_safe_error_type_and_code(tmp_path, monkeypatch):
    failure = HTTPError("https://secret-key.example", 401, "secret-key", {}, io.BytesIO(b"secret-key"))
    sent = transport(monkeypatch, iter([failure]))
    with pytest.raises(ModelError, match="HTTP 401"):
        complete(config(), "inspect", trace_dir=tmp_path)
    assert len(sent) == 1
    assert load(tmp_path / "error.json") == {"type": "HTTPError", "http_code": 401}
    assert load(tmp_path / "metadata.json")["status"] == "failed"
    assert "secret-key" not in "".join(p.read_text() for p in tmp_path.iterdir())


def test_http_error_trace_keeps_diagnostic_fields_and_redacts_secrets(tmp_path, monkeypatch):
    body = {"error": {"type": "invalid_request_error", "code": "unsupported_value", "param": "temperature",
        "message": "temperature must be 1; secret-key at https://private.example/v1; "
                   "https://other.example/path?key=hidden Authorization: Bearer other-token "
                   "api_key=additional-key password=private-password X-Private: hidden-header "
                   "headers={'Another-Header': 'embedded-header-secret'}",
        "headers": {"Authorization": "unknown-secret"}}, "debug": "debug-secret"}
    failure = HTTPError("https://private.example/v1", 400, "invalid", {"X-Secret": "header-secret"},
                        io.BytesIO(json.dumps(body).encode()))
    transport(monkeypatch, iter([failure]))
    with pytest.raises(ModelError, match="HTTP 400"):
        complete(config(), "inspect", trace_dir=tmp_path)
    diagnostic = load(tmp_path / "http_error.json")
    assert diagnostic["representation"] == "sanitized_fields_not_raw_body"
    assert diagnostic["error"]["param"] == "temperature"
    assert diagnostic["error"]["code"] == "unsupported_value"
    assert "temperature must be 1" in diagnostic["error"]["message"]
    encoded = json.dumps(diagnostic)
    for private in ("secret-key", "private.example", "other.example", "other-token", "additional-key",
                    "private-password", "unknown-secret", "debug-secret", "header-secret", "Authorization",
                    "hidden-header", "embedded-header-secret", "Another-Header"):
        assert private not in encoded
    assert set(diagnostic["error"]) == {"type", "code", "param", "message"}


def test_non_json_http_error_trace_is_bounded_and_redacted(tmp_path, monkeypatch):
    body = b"Unsupported temperature; secret-key https://private.example/v1 " + b"x" * 10000
    transport(monkeypatch, iter([HTTPError("https://private.example", 400, "bad", {}, io.BytesIO(body))]))
    with pytest.raises(ModelError):
        complete(config(), "inspect", trace_dir=tmp_path)
    diagnostic = load(tmp_path / "http_error.json")
    assert diagnostic["truncated"]
    assert len(diagnostic["error"]["message"]) <= 2048
    assert "Unsupported temperature" in diagnostic["error"]["message"]
    assert "secret-key" not in json.dumps(diagnostic)
    assert "private.example" not in json.dumps(diagnostic)


def test_default_http_error_does_not_read_body(monkeypatch):
    class UnreadBody(io.BytesIO):
        def read(self, *args):
            pytest.fail("default adapter must not read the HTTP error body")

    transport(monkeypatch, iter([HTTPError("https://private.example", 400, "bad", {}, UnreadBody())]))
    with pytest.raises(ModelError, match="HTTP 400"):
        complete(config(), "inspect")


def test_shared_scope_limit_image_copy_and_real_usage(tmp_path, monkeypatch):
    photo = tmp_path / "input.png"
    Image.new("RGB", (4, 4)).save(photo)
    sent = transport(monkeypatch, iter([response(), response()]))
    output = tmp_path / "run"
    api = FrozenAPI(config(), output, max_calls=2)
    api.scoped("generator")("generate", [photo])
    api.scoped("active/judge")("diagnose")
    with pytest.raises(ModelError, match="request-count limit"):
        api("extra")
    assert len(sent) == 2
    ledger = load(output / "ledger.json")
    assert ledger["request_count"] == 2
    assert ledger["returned_model"] == "served-v1"
    assert ledger["config"]["role"] == "shared"
    assert not ledger["dollar_cap_enforced"]
    assert [c["scope"] for c in ledger["calls"]] == ["generator", "active/judge"]
    assert ledger["calls"][0]["usage"]["total_tokens"] == 20
    assert ledger["calls"][0]["provider_cost"] is None
    copied = output / "api_calls/001/image_000.png"
    assert copied.read_bytes() == photo.read_bytes()
    assert len(load(output / "api_calls/001/images.json")[0]["sha256"]) == 64
    with pytest.raises(ValueError, match="already contains"):
        FrozenAPI(config(), output)


def test_count_is_written_before_request_and_failure_stops_run(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        assert load(tmp_path / "ledger.json")["request_count"] == 1
        raise ModelError("failure includes secret-key")

    monkeypatch.setattr("turnitover.api_pipeline.api.complete", fail)
    api = FrozenAPI(config(), tmp_path)
    with pytest.raises(ModelError):
        api("generate")
    ledger = load(tmp_path / "ledger.json")
    assert ledger["calls"][0]["status"] == "failed"
    assert ledger["stopped"]
    assert "secret-key" not in json.dumps(ledger)
    with pytest.raises(ModelError, match="previous failure"):
        api("retry")
    assert len(api.calls) == 1


@pytest.mark.parametrize("models,expected", [(["served-v1", "served-v2"], None), (["served-v2"], "served-v1")])
def test_returned_model_identity_is_frozen_across_calls_and_runs(tmp_path, monkeypatch, models, expected):
    sent = transport(monkeypatch, iter(response(model) for model in models))
    api = FrozenAPI(config(), tmp_path, expected_model=expected)
    if len(models) == 2:
        api("first")
    with pytest.raises(ModelError, match="different model identity"):
        api("changed")
    assert api.returned_model == "served-v1"
    assert api.calls[-1]["status"] == "failed"
    with pytest.raises(ModelError, match="previous failure"):
        api("retry")
    assert len(sent) == len(models)
