"""Offline SSE wire replay, completion checks, and frozen API transport recording."""
import io
import json

import pytest

from turnitover.api_pipeline.api import FrozenAPI
from turnitover.models.client import ModelError, complete
from turnitover.models.config import ModelConfig


def config(provider="openai_compatible"):
    return ModelConfig("teacher", provider, "k3-256k", "https://private.example/v1", "secret-key")


def event(delta=None, *, finish=None, usage=None):
    result = {"model": "k3-pinned", "choices": []}
    if delta is not None or finish is not None:
        result["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        result["usage"] = usage
    return result


def sse(chunks, done=True):
    wire = b": keepalive\r\n\r\n"
    wire += b"".join(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\r\n\r\n" for chunk in chunks)
    return wire + (b"data: [DONE]\r\n\r\n" if done else b"")


def transport(monkeypatch, wire):
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(wire)

    monkeypatch.setattr("turnitover.models.client.build_opener", lambda *args: Opener())
    return requests


@pytest.mark.parametrize("provider", ["openai_compatible", "openrouter"])
def test_real_wire_assembles_content_reasoning_finish_and_exact_usage(tmp_path, monkeypatch, provider):
    usage = {"prompt_tokens": 42, "completion_tokens": 9, "completion_tokens_details": {"reasoning_tokens": 5}}
    chunks = [event({"role": "assistant"}), event({"reasoning_content": "think "}),
              event({"reasoning_content": "carefully"}), event({"content": "真实"}),
              event({"content": " answer"}), event(finish="stop"), event(usage=usage)]
    wire = sse(chunks)
    sent = transport(monkeypatch, wire)
    result = complete(config(provider), "inspect", trace_dir=tmp_path, stream=True)
    assert result.text == "真实 answer" and "think" not in result.text
    assert result.model == "k3-pinned" and result.usage == usage
    assert sent[0]["stream"] is True
    assert sent[0]["stream_options"] == {"include_usage": True}
    assert "temperature" not in sent[0]
    assert (tmp_path / "response.sse").read_bytes() == wire
    assembled = json.loads((tmp_path / "response.json").read_text())
    assert assembled["choices"][0]["message"] == {
        "role": "assistant", "content": "真实 answer", "reasoning_content": "think carefully"}
    assert assembled["usage"] == chunks[-1]["usage"]
    assert json.loads((tmp_path / "metadata.json").read_text())["transport"] == {"stream": True}


@pytest.mark.parametrize("chunks,done", [
    ([event({"content": "partial"})], False),
    ([event({"content": "partial"}), event(finish="stop")], False),
    ([event({"content": "partial"})], True),
])
def test_truncation_never_becomes_success(tmp_path, monkeypatch, chunks, done):
    wire = sse(chunks, done)
    transport(monkeypatch, wire)
    with pytest.raises(ModelError, match="Incomplete SSE response"):
        complete(config(), "inspect", trace_dir=tmp_path, stream=True)
    assert (tmp_path / "response.sse").read_bytes() == wire
    assert not (tmp_path / "response.json").exists()
    assert json.loads((tmp_path / "error.json").read_text())["type"] == "StreamingError"


def test_api_error_chunk_stops_and_preserves_received_wire(tmp_path, monkeypatch):
    wire = sse([event({"reasoning_content": "thinking"}), {"error": {"message": "secret-key"}}], False)
    transport(monkeypatch, wire)
    with pytest.raises(ModelError, match="SSE error chunk") as caught:
        complete(config(), "inspect", trace_dir=tmp_path, stream=True)
    assert "secret-key" not in str(caught.value)
    assert b'"error"' in (tmp_path / "response.sse").read_bytes()
    assert not (tmp_path / "response.json").exists()


def test_missing_usage_is_not_fabricated(tmp_path, monkeypatch):
    transport(monkeypatch, sse([event({"content": "ok"}), event(finish="stop")]))
    result = complete(config(), "inspect", trace_dir=tmp_path, stream=True)
    assert result.usage == {}
    assert "usage" not in json.loads((tmp_path / "response.json").read_text())


def test_frozen_api_records_transport_separately_from_sampling(tmp_path, monkeypatch):
    transport(monkeypatch, sse([event({"content": "ok"}), event(finish="stop")]))
    api = FrozenAPI(config(), tmp_path, stream=True)
    api("inspect")
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["transport"] == {"stream": True}
    assert "stream" not in ledger["config"]
    assert (tmp_path / "api_calls/001/response.sse").exists()


def test_other_providers_reject_stream_before_request(tmp_path, monkeypatch):
    monkeypatch.setattr("turnitover.models.client.build_opener", lambda *args: pytest.fail("must not send request"))
    with pytest.raises(ValueError, match="only for compatible"):
        complete(config("gemini"), "inspect", trace_dir=tmp_path, stream=True)
