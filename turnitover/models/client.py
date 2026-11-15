"""Small REST adapters for text + images. No implicit retries (TransientModelError lets callers retry) or SDK dependencies."""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image

from turnitover.models.config import ModelConfig


# Finish reasons of a normally finished response: OpenAI Responses, OpenAI-compatible chat, Anthropic, Gemini.
FINISHED = frozenset({"completed", "stop", "end_turn", "stop_sequence", "STOP"})
TRANSIENT_HTTP = frozenset({429, 500, 502, 503, 504})


class ModelError(RuntimeError):
    pass


class TransientModelError(ModelError):
    """Rate limit, overload or dropped connection: the same request may succeed later. The caller decides on retries."""

    def __init__(self, provider: str, reason: str, retry_after: float | None = None):
        super().__init__(f"{provider} request failed transiently: {reason}")
        self.reason = reason
        self.retry_after = retry_after


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


def image_part(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        mime = Image.MIME.get(image.format)
        if mime not in {"image/png", "image/jpeg", "image/webp"} or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Model images must be static PNG, JPEG or WebP; extract video frames first")
        image.verify()
    return mime, base64.b64encode(path.read_bytes()).decode("ascii")


def request_payload(cfg: ModelConfig, prompt: str, images: list[Path]) -> tuple[str, dict, dict]:
    parts = [image_part(path) for path in images]
    # Relays behind Cloudflare reject the default urllib user agent (error 1010).
    headers = {"Content-Type": "application/json", "User-Agent": "turnitover-model-client/1.0"}
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
    content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}} for mime, data in parts]
    content.append({"type": "text", "text": prompt})
    body = {"model": cfg.model, "max_tokens": cfg.max_tokens, "messages": [{"role": "user", "content": content}]}
    if cfg.chat_template_kwargs is not None:
        body["chat_template_kwargs"] = cfg.chat_template_kwargs
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
            raise ModelError("Provider returned no text (possibly refusal, token limit or incompatible model)")
        return ModelResult(text, data.get("model", data.get("modelVersion", "")),
                           data.get("usage", data.get("usageMetadata", {})), finish, elapsed_ms)
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ModelError("Unexpected provider response schema") from None


def retry_after_seconds(value: str | None) -> float | None:
    """Retry-After header: delta-seconds or an HTTP date."""
    if value is None:
        return None
    if value.strip().isdigit():
        return float(value)
    return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())


def complete(cfg: ModelConfig, prompt: str, images: list[Path] | None = None) -> ModelResult:
    cfg.validate()
    url, headers, payload = request_payload(cfg, prompt, images or [])
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with build_opener(_NoRedirect()).open(request, timeout=cfg.timeout) as response:
            data = json.load(response)
    except HTTPError as exc:
        # Never echo headers, URLs or provider error bodies (they can contain credentials).
        if exc.code in TRANSIENT_HTTP:
            raise TransientModelError(cfg.provider, f"HTTP {exc.code}",
                                      retry_after_seconds(exc.headers.get("Retry-After"))) from None
        raise ModelError(f"{cfg.provider} request failed with HTTP {exc.code}") from None
    except (OSError, IncompleteRead) as exc:   # URLError, timeouts, connection resets; type names only, never the URL
        reason = type(exc).__name__ + (f"/{type(exc.reason).__name__}" if isinstance(exc, URLError) else "")
        raise TransientModelError(cfg.provider, reason) from None
    except (ValueError, UnicodeError):
        raise ModelError("Provider returned invalid JSON") from None
    return parse_response(cfg.provider, data, (time.perf_counter() - start) * 1000)
