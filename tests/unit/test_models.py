import io
import json
from urllib.error import HTTPError

import pytest
from PIL import Image

from turnitover.models.client import ModelError, complete, parse_response, request_payload
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


@pytest.mark.parametrize("provider,suffix,image_key,token_key", [
    ("openai", "/responses", "input_image", "max_output_tokens"),
    ("anthropic", "/messages", "media_type", "max_tokens"),
    ("gemini", "/models/test-model:generateContent", "inlineData", "maxOutputTokens"),
    ("openai_compatible", "/chat/completions", "image_url", "max_tokens"),
    ("openrouter", "/chat/completions", "image_url", "max_tokens"),
])
def test_provider_vision_wire_formats(tmp_path, provider, suffix, image_key, token_key):
    path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), "blue").save(path)
    cfg = ModelConfig("generator", provider, "test-model", "https://test/v1", "secret-key")
    url, headers, body = request_payload(cfg, "Describe this", [path])
    assert url.endswith(suffix)
    encoded = json.dumps(body)
    assert image_key in encoded and token_key in encoded
    assert "secret-key" not in encoded + url
    assert "secret-key" in json.dumps(headers)


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
