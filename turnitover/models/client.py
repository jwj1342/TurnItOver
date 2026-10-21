"""Small REST adapters for text + images. No implicit retries or SDK dependencies."""
from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image

from turnitover.models.config import ModelConfig
from turnitover.models.streaming import StreamingError, read_chat_stream


class ModelError(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class ModelResult:
    text: str
    model: str
    usage: dict
    finish_reason: str
    elapsed_ms: float


NORMAL_FINISH_REASONS = frozenset({"completed", "end_turn", "STOP", "stop"})


def is_normal_finish_reason(reason: str) -> bool:
    return reason in NORMAL_FINISH_REASONS


def image_part(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        mime = Image.MIME.get(image.format)
        if mime not in {"image/png", "image/jpeg", "image/webp"} or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Model images must be static PNG, JPEG or WebP; extract video frames first")
        image.verify()
    return mime, base64.b64encode(path.read_bytes()).decode("ascii")


def request_payload(cfg: ModelConfig, prompt: str, images: list[Path]) -> tuple[str, dict, dict]:
    parts = [image_part(path) for path in images]
    headers = {"Content-Type": "application/json"}
    if cfg.provider == "anthropic":
        headers.update({"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"})
        content = [{"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
                   for mime, data in parts] + [{"type": "text", "text": prompt}]
        return cfg.base_url + "/messages", headers, {"model": cfg.model, "max_tokens": cfg.max_tokens,
                                                    "messages": [{"role": "user", "content": content}]}
    if cfg.provider == "gemini":
        headers["x-goog-api-key"] = cfg.api_key
        content = [{"inlineData": {"mimeType": mime, "data": data}} for mime, data in parts]
        content.append({"text": prompt})
        model = quote(cfg.model.removeprefix("models/"), safe="")
        return cfg.base_url + f"/models/{model}:generateContent", headers, {
            "contents": [{"role": "user", "parts": content}],
            "generationConfig": {"maxOutputTokens": cfg.max_tokens}}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if cfg.provider == "openai":
        content = [{"type": "input_image", "image_url": f"data:{mime};base64,{data}"} for mime, data in parts]
        content.append({"type": "input_text", "text": prompt})
        return cfg.base_url + "/responses", headers, {"model": cfg.model, "store": False,
            "max_output_tokens": cfg.max_tokens, "input": [{"role": "user", "content": content}]}
    if cfg.provider == "openai_compatible":
        headers["User-Agent"] = "TurnItOver/1.0 OpenAI-Compatible-Client"
    content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}} for mime, data in parts]
    content.append({"type": "text", "text": prompt})
    body = {"model": cfg.model, "max_tokens": cfg.max_tokens,
            "messages": [{"role": "user", "content": content}]}
    if (cfg.provider == "openai_compatible" and cfg.model.startswith("qwen3.7-plus")
            and cfg.enable_thinking is not None):
        body["enable_thinking"] = cfg.enable_thinking
    return cfg.base_url + "/chat/completions", headers, body


def parse_response(provider: str, data: dict, elapsed_ms: float) -> ModelResult:
    try:
        if provider == "openai":
            text = "\n".join(c["text"] for item in data.get("output", []) if item.get("type") == "message"
                             for c in item.get("content", []) if c.get("type") == "output_text")
            finish = data.get("status", "unknown")
        elif provider == "anthropic":
            text = "\n".join(c["text"] for c in data.get("content", []) if c.get("type") == "text")
            finish = data.get("stop_reason", "unknown")
        elif provider == "gemini":
            candidate = data["candidates"][0]
            text = "\n".join(c["text"] for c in candidate.get("content", {}).get("parts", [])
                             if "text" in c and not c.get("thought"))
            finish = candidate.get("finishReason", "unknown")
        else:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            finish = choice.get("finish_reason", "unknown")
        if not isinstance(text, str) or not text.strip():
            usage = data.get("usage", data.get("usageMetadata", {}))
            raise ModelError("Provider returned no text (possibly refusal, token limit or incompatible model); "
                             f"finish_reason={finish}; usage={json.dumps(usage, sort_keys=True)}")
        return ModelResult(text, data.get("model", data.get("modelVersion", "")),
                           data.get("usage", data.get("usageMetadata", {})), finish, elapsed_ms)
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ModelError("Unexpected provider response schema") from None


def _trace_json(directory: Path | None, name: str, value: dict) -> None:
    if directory is not None:
        (directory / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")


def _http_error_diagnostic(exc: HTTPError, cfg: ModelConfig) -> dict:
    """Bounded, sanitized provider fields, never raw bytes or response headers."""
    try:
        raw = exc.read(8193)
    except Exception:
        return {"representation": "sanitized_fields_not_raw_body", "body_unavailable": True}
    text = raw[:8192].decode("utf-8", errors="replace")
    try:
        body = json.loads(text)
    except ValueError:
        fields = {"message": text}
    else:
        error = body.get("error", {}) if isinstance(body, dict) else {}
        fields = {key: error[key] for key in ("type", "code", "param", "message")
                  if isinstance(error, dict) and key in error
                  and isinstance(error[key], (str, int, float, bool, type(None)))}

    def redact(value):
        if not isinstance(value, str):
            return value
        for secret in (cfg.api_key, cfg.base_url):
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)\b(?:https?|wss?)://[^\s<>\"']+", "[REDACTED_URL]", value)
        value = re.sub(r"(?is)\b(?:request[_ ]?|response[_ ]?)?headers\s*[:=]\s*"
                       r"(?:\{.*?\}|\[.*?\]|[^\r\n]*)", "[REDACTED_HEADERS]", value)
        value = re.sub(
            r"(?i)\b(?:authorization|proxy-authorization|x-[a-z0-9-]+|api[_ -]?key|"
            r"set-cookie|cookie|password|secret|token)[\"']?\s*[:=]\s*[\"']?"
            r"(?:bearer\s+)?[^\s,;\"']+", "[REDACTED]", value)
        value = re.sub(r"(?i)\bbearer\s+[^\s,;\"']+", "[REDACTED]", value)
        value = re.sub(r"\b(?:sk|rk)-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        return value[:2048]

    return {"representation": "sanitized_fields_not_raw_body", "truncated": len(raw) > 8192,
            "error": {key: redact(value) for key, value in fields.items()}}


def complete(cfg: ModelConfig, prompt: str, images: list[Path] | None = None, *,
             trace_dir: Path | None = None, temperature: float | None = None,
             stream: bool = False) -> ModelResult:
    """Send once; optional traces contain payloads but never endpoint or headers."""
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    data = {}
    result = None
    failure = None
    request = None
    try:
        cfg.validate()
        if stream and cfg.provider not in {"openai_compatible", "openrouter"}:
            raise ValueError("SSE transport is supported only for compatible Chat Completions providers")
        url, headers, payload = request_payload(cfg, prompt, images or [])
        if stream:
            payload.update(stream=True, stream_options={"include_usage": True})
        if temperature is not None:
            if cfg.provider == "gemini":
                payload["generationConfig"]["temperature"] = temperature
            else:
                payload["temperature"] = temperature
        _trace_json(trace_dir, "request.json", payload)
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        start = time.perf_counter()
        with build_opener(_NoRedirect()).open(request, timeout=cfg.timeout) as response:
            data = read_chat_stream(response, trace_dir / "response.sse" if trace_dir else None) if stream else json.load(response)
        _trace_json(trace_dir, "response.json", data)
        result = parse_response(cfg.provider, data, (time.perf_counter() - start) * 1000)
        return result
    except StreamingError as exc:
        failure = {"type": type(exc).__name__, "http_code": None}
        raise ModelError(str(exc)) from None
    except HTTPError as exc:
        # No response body in exceptions; opt-in traces retain only sanitized fields.
        failure = {"type": type(exc).__name__, "http_code": exc.code}
        if trace_dir is not None:
            _trace_json(trace_dir, "http_error.json", _http_error_diagnostic(exc, cfg))
        raise ModelError(f"{cfg.provider} request failed with HTTP {exc.code}; no automatic retry") from None
    except (URLError, TimeoutError, OSError) as exc:
        failure = {"type": type(exc).__name__, "http_code": None}
        if request is None:
            raise
        raise ModelError(f"{cfg.provider} connection failed or timed out; no automatic retry") from None
    except (ValueError, UnicodeError) as exc:
        failure = {"type": type(exc).__name__, "http_code": None}
        # Preserve validation errors before constructing a request.
        if request is None:
            raise
        raise ModelError("Provider returned invalid JSON") from None
    except Exception as exc:
        failure = {"type": type(exc).__name__, "http_code": None}
        raise
    finally:
        if failure is not None:
            _trace_json(trace_dir, "error.json", failure)
        usage = data.get("usage", data.get("usageMetadata", {})) if isinstance(data, dict) else {}
        provider_cost = usage.get("cost") if isinstance(usage, dict) else None
        if provider_cost is None and isinstance(data, dict):
            provider_cost = data.get("cost")
        _trace_json(trace_dir, "metadata.json", {
            "provider": cfg.provider, "requested_model": cfg.model,
            "transport": {"stream": stream},
            "returned_model": result.model if result else None,
            "usage": usage, "provider_cost": provider_cost,
            "elapsed_ms": (time.perf_counter() - start) * 1000,
            "finish_reason": result.finish_reason if result else None,
            "status": "complete" if result else "failed",
        })
