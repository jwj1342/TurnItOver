"""Small REST adapters for text + images. No implicit retries or SDK dependencies."""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image

from turnitover.models.config import ModelConfig


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
    content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}} for mime, data in parts]
    content.append({"type": "text", "text": prompt})
    return cfg.base_url + "/chat/completions", headers, {"model": cfg.model, "max_tokens": cfg.max_tokens,
        "messages": [{"role": "user", "content": content}]}


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
        raise ModelError(f"{cfg.provider} request failed with HTTP {exc.code}; no automatic retry") from None
    except (URLError, TimeoutError, OSError):
        raise ModelError(f"{cfg.provider} connection failed or timed out; no automatic retry") from None
    except (ValueError, UnicodeError):
        raise ModelError("Provider returned invalid JSON") from None
    return parse_response(cfg.provider, data, (time.perf_counter() - start) * 1000)
