import base64
import io
import json
from urllib.error import HTTPError, URLError

import pytest
from PIL import Image

from turnitover.models.client import (FINISHED, ModelError, TransientModelError, complete, parse_response,
                                      request_payload)
from turnitover.models.commands import extract_program
from turnitover.models.config import ModelConfig, model_config, read_environment


def test_env_precedence_quotes_and_no_shell_execution(tmp_path):
    path = tmp_path / ".env"
    path.write_text('export KEY="file value" # comment\nEMPTY=\nLITERAL=\'$(echo nope)\'\n')
    env = read_environment(path, {"KEY": "process value"})
    assert env == {"KEY": "process value", "EMPTY": "", "LITERAL": "$(echo nope)"}


def test_env_parse_error_does_not_echo_secret(tmp_path):
    path = tmp_path / ".env"
    path.write_text('KEY="secret-key')
    with pytest.raises(ValueError) as caught:
        read_environment(path, {})
    assert "secret-key" not in str(caught.value)


def test_roles_overrides_and_secret_redaction():
    env = {"TIO_GENERATOR_PROVIDER": "gemini", "TIO_GENERATOR_MODEL": "my-model",
           "TIO_GENERATOR_API_KEY_ENV": "SPECIAL", "SPECIAL": "secret-key",
           "TIO_GENERATOR_MAX_TOKENS": "8000"}
    cfg = model_config("generator", env)
    cfg.validate()
    assert cfg.api_key == "secret-key" and cfg.max_tokens == 8000
    assert "secret-key" not in repr(cfg) + json.dumps(cfg.public())
    assert model_config("judge", env).model == ""


@pytest.mark.parametrize("url", ["https://u:secret@host/v1", "https://host/v1?key=secret", "http://remote/v1"])
def test_reject_credential_urls_and_remote_plaintext(url):
    with pytest.raises(ValueError):
        model_config("generator", {"TIO_GENERATOR_BASE_URL": url})


@pytest.mark.parametrize("provider,suffix", [
    ("openai", "/responses"), ("anthropic", "/messages"),
    ("gemini", "/models/test-model:generateContent"),
    ("openai_compatible", "/chat/completions"),
])
def test_provider_vision_wire_formats(tmp_path, provider, suffix):
    path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), "blue").save(path)
    cfg = ModelConfig("generator", provider, "test-model", "https://test/v1", "secret-key", max_tokens=321)
    url, headers, body = request_payload(cfg, "Describe this", [path])
    assert url == cfg.base_url + suffix
    data = base64.b64encode(path.read_bytes()).decode()
    data_url = f"data:image/png;base64,{data}"
    if provider == "gemini":
        assert body["generationConfig"]["maxOutputTokens"] == 321
        assert body["contents"] == [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/png", "data": data}}, {"text": "Describe this"}]}]
        assert headers["x-goog-api-key"] == "secret-key"
    else:
        assert body["model"] == "test-model"
        if provider == "openai":
            assert body["max_output_tokens"] == 321 and body["store"] is False
            assert body["input"] == [{"role": "user", "content": [
                {"type": "input_image", "image_url": data_url}, {"type": "input_text", "text": "Describe this"}]}]
        else:
            assert body["max_tokens"] == 321
            image = ({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
                     if provider == "anthropic" else {"type": "image_url", "image_url": {"url": data_url}})
            assert body["messages"] == [{"role": "user", "content": [image, {"type": "text", "text": "Describe this"}]}]
        if provider == "anthropic":
            assert headers["x-api-key"] == "secret-key"
        else:
            assert headers["Authorization"] == "Bearer secret-key"
    assert "secret-key" not in json.dumps(body) + url


@pytest.mark.parametrize("provider,response", [
    ("openai", {"output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}], "status": "completed"}),
    ("anthropic", {"content": [{"type": "thinking", "thinking": "private"}, {"type": "text", "text": "ok"}], "stop_reason": "end_turn"}),
    ("gemini", {"candidates": [{"content": {"parts": [{"thought": True, "text": "private"}, {"text": "ok"}]}, "finishReason": "STOP"}]}),
    ("openai_compatible", {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}),
])
def test_response_text_excludes_reasoning(provider, response):
    assert parse_response(provider, response, 1).text == "ok"


def test_http_error_redacts_body_and_url(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            raise HTTPError("https://secret-key.example", 401, "secret-key", {}, io.BytesIO(b"secret-key"))

    monkeypatch.setattr("turnitover.models.client.build_opener", lambda *a: Opener())
    cfg = ModelConfig("generator", "openai", "test", "https://test/v1", "secret-key")
    with pytest.raises(ModelError) as caught:
        complete(cfg, "hello")
    assert "401" in str(caught.value) and "secret-key" not in str(caught.value)


def test_mock_transport_end_to_end(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            assert json.loads(request.data)["model"] == "test"
            return io.BytesIO(json.dumps({"model": "test-pinned", "status": "completed", "usage": {"input_tokens": 12},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}]}).encode())

    monkeypatch.setattr("turnitover.models.client.build_opener", lambda *a: Opener())
    result = complete(ModelConfig("generator", "openai", "test", "https://test/v1", "key"), "hi")
    assert result.model == "test-pinned" and result.usage["input_tokens"] == 12


def test_vllm_generator_request_and_env_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TIO_GENERATOR_PROVIDER=openai\nTIO_GENERATOR_MODEL=other\nCOMPATIBLE_BASE_URL=http://localhost:1/v1\n")
    env = read_environment(env_file, {"TIO_GENERATOR_PROVIDER": "openai_compatible",
                                      "TIO_GENERATOR_MODEL": "Qwen/Qwen3.8-27B-FP8",
                                      "COMPATIBLE_BASE_URL": "http://gpu-node-7:8000/v1",
                                      "TIO_GENERATOR_MAX_TOKENS": "16384",
                                      "TIO_GENERATOR_CHAT_TEMPLATE_KWARGS": '{"enable_thinking": false}'})
    cfg = model_config("generator", env)
    cfg.validate()
    assert cfg.public()["chat_template_kwargs"] == {"enable_thinking": False}
    images = []
    for name, colour in (("reference.png", "red"), ("candidate.png", "blue")):
        images.append(tmp_path / name)
        Image.new("RGB", (4, 4), colour).save(images[-1])
    seen = {}
    class Opener:
        def open(self, request, timeout):
            seen.update(url=request.full_url, body=json.loads(request.data), headers=dict(request.header_items()))
            return io.BytesIO(json.dumps({"model": "Qwen/Qwen3.8-27B-FP8", "usage": {"total_tokens": 99},
                "choices": [{"message": {"content": '{"stop": true}', "reasoning_content": "hidden"},
                             "finish_reason": "stop"}]}).encode())

    monkeypatch.setattr("turnitover.models.client.build_opener", lambda *a: Opener())
    result = complete(cfg, "repair this", images)
    assert seen["url"] == "http://gpu-node-7:8000/v1/chat/completions"
    assert "Authorization" not in seen["headers"]
    body = seen["body"]
    assert body["model"] == "Qwen/Qwen3.8-27B-FP8" and body["max_tokens"] == 16384
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    urls = [part["image_url"]["url"] for part in body["messages"][0]["content"] if part["type"] == "image_url"]
    assert urls == [f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" for p in images]
    assert body["messages"][0]["content"][-1] == {"type": "text", "text": "repair this"}
    assert result.text == '{"stop": true}' and result.finish_reason in FINISHED and result.usage["total_tokens"] == 99
    plain = model_config("generator", {**env, "TIO_GENERATOR_CHAT_TEMPLATE_KWARGS": ""})
    assert "chat_template_kwargs" not in request_payload(plain, "x", [])[2]
    assert model_config("generator", {**env, "TIO_GENERATOR_PROVIDER": "openrouter"}).chat_template_kwargs is None


@pytest.mark.parametrize("error,reason,retry_after", [
    (HTTPError("https://secret-key.example", 503, "secret-key", {"Retry-After": "7"}, io.BytesIO(b"secret-key")), "HTTP 503", 7.0),
    (HTTPError("https://secret-key.example", 429, "secret-key", {}, io.BytesIO(b"secret-key")), "HTTP 429", None),
    (URLError(ConnectionRefusedError("secret-key.example")), "URLError/ConnectionRefusedError", None),
    (TimeoutError("timed out"), "TimeoutError", None),
    (ConnectionResetError("reset"), "ConnectionResetError", None),
])
def test_transient_failures_are_classified_without_leaking(monkeypatch, error, reason, retry_after):
    class Opener:
        def open(self, request, timeout):
            raise error

    monkeypatch.setattr("turnitover.models.client.build_opener", lambda *a: Opener())
    with pytest.raises(TransientModelError) as caught:
        complete(ModelConfig("generator", "openai_compatible", "m", "http://node:8000/v1", "secret-key"), "hi")
    assert caught.value.reason == reason and caught.value.retry_after == retry_after
    assert "secret-key" not in str(caught.value)


def test_other_client_errors_are_not_transient(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            raise HTTPError("https://x", 400, "bad", {}, io.BytesIO(b""))

    monkeypatch.setattr("turnitover.models.client.build_opener", lambda *a: Opener())
    with pytest.raises(ModelError) as caught:
        complete(ModelConfig("generator", "openai_compatible", "m", "http://node:8000/v1", ""), "hi")
    assert not isinstance(caught.value, TransientModelError) and "HTTP 400" in str(caught.value)


def test_reconstruct_dry_run_never_calls_api(tmp_path, monkeypatch):
    from turnitover.cli import main

    image = tmp_path / "photo.png"
    Image.new("RGB", (8, 8)).save(image)
    monkeypatch.setattr("turnitover.models.commands.complete", lambda *a: pytest.fail("dry run called API"))
    output = tmp_path / "prepared"
    assert main(["reconstruct", "--env-file", str(tmp_path / "absent"), "--image", str(image),
                 "--out", str(output), "--dry-run"]) == 0
    assert json.loads((output / "manifest.json").read_text())["status"] == "prepared"
    assert not (output / "program.ts").exists()


def test_program_extraction_rejects_ambiguous_outputs():
    assert extract_program('```ts\nexport default function() {}\n```').startswith("export default")
    with pytest.raises(ValueError):
        extract_program("This request was refused")
    with pytest.raises(ValueError):
        extract_program('```ts\nexport default 1\n```\n```ts\nexport default 2\n```')


@pytest.mark.parametrize("finish,expected_status", [("completed", "complete"), ("incomplete", "failed")])
def test_reconstruction_records_outcome_and_rejects_truncation(tmp_path, monkeypatch, finish, expected_status):
    from turnitover.cli import main
    from turnitover.models.client import ModelResult

    photo = tmp_path / "photo.png"
    Image.new("RGB", (4, 4)).save(photo)
    monkeypatch.setenv("TIO_GENERATOR_PROVIDER", "openai")
    monkeypatch.setenv("TIO_GENERATOR_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    source = "export default function createObject(THREE) { return {root:new THREE.Group(), joints:{}}; }"
    monkeypatch.setattr("turnitover.models.commands.complete",
                        lambda *a: ModelResult(source, "test-version", {"input_tokens": 2}, finish, 10))
    compiled = []
    monkeypatch.setattr("turnitover.render.transpile.transpile_ts", lambda text, path: compiled.append(text))
    out = tmp_path / "result"
    argv = ["reconstruct", "--env-file", str(tmp_path / "none"), "--image", str(photo), "--out", str(out)]
    if finish == "incomplete":
        with pytest.raises(ValueError, match="did not finish"):
            main(argv)
        assert not compiled and not (out / "program.ts").exists()
    else:
        assert main(argv) == 0
        assert compiled and (out / "program.ts").exists()
    metadata = (out / "manifest.json").read_text()
    assert json.loads(metadata)["status"] == expected_status
    assert "test-secret" not in metadata
    assert (out / "response.txt").exists()
