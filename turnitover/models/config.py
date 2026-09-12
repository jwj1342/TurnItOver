"""Explicit model selection and a non-executing, intentionally small .env reader."""
from __future__ import annotations

import math
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ROLES = ("generator", "judge", "diagnosis")
PROVIDERS = {
    "openai": ("OPENAI", "https://api.openai.com/v1"),
    "anthropic": ("ANTHROPIC", "https://api.anthropic.com/v1"),
    "gemini": ("GEMINI", "https://generativelanguage.googleapis.com/v1beta"),
    "openai_compatible": ("COMPATIBLE", "http://localhost:8000/v1"),
    "openrouter": ("OPENROUTER", "https://openrouter.ai/api/v1"),
}


def read_environment(path: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Read KEY=value, quotes, comments, optional export. No shell/variable expansion."""
    values = {}
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, sep, value = line.partition("=")
            key = key.strip()
            if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid .env assignment at line {number}")
            try:
                tokens = shlex.split(value, comments=True, posix=True)
            except ValueError:
                raise ValueError(f"Invalid .env quoting at line {number}") from None
            if len(tokens) > 1:
                raise ValueError(f"Quote .env values containing spaces (line {number})")
            values[key] = tokens[0] if tokens else ""
    values.update(os.environ if environ is None else environ)
    return values


@dataclass(frozen=True)
class ModelConfig:
    role: str
    provider: str
    model: str
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)
    max_tokens: int = 4096
    timeout: float = 120

    def public(self) -> dict:
        # Endpoint URLs can contain private routing information; do not log them.
        return {"role": self.role, "provider": self.provider, "model": self.model,
                "key_configured": bool(self.api_key), "max_tokens": self.max_tokens,
                "timeout": self.timeout}

    def validate(self, require_key: bool = True) -> None:
        if not self.model:
            raise ValueError(f"Set TIO_{self.role.upper()}_MODEL to an explicit model ID")
        if require_key and not self.api_key and self.provider != "openai_compatible":
            raise ValueError(f"API key is missing for {self.role} ({self.provider})")


def model_config(role: str, env: dict[str, str]) -> ModelConfig:
    if role not in ROLES:
        raise ValueError(f"Unknown model role: {role}")
    prefix = f"TIO_{role.upper()}"
    provider = env.get(f"{prefix}_PROVIDER", "openai")
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider for {role}")
    provider_prefix, default_url = PROVIDERS[provider]
    base = env.get(f"{prefix}_BASE_URL") or env.get(f"{provider_prefix}_BASE_URL") or default_url
    url = urlsplit(base)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        raise ValueError("Base URL must be an HTTP(S) endpoint without credentials, query or fragment")
    if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Remote model endpoints require HTTPS; use a localhost tunnel for local servers")
    key_name = env.get(f"{prefix}_API_KEY_ENV") or f"{provider_prefix}_API_KEY"
    try:
        limit = int(env.get(f"{prefix}_MAX_TOKENS") or env.get("TIO_MODEL_MAX_TOKENS") or "4096")
        timeout = float(env.get(f"{prefix}_TIMEOUT") or env.get("TIO_MODEL_TIMEOUT") or "120")
    except ValueError:
        raise ValueError("Model token limit and timeout must be numeric") from None
    if limit <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Model token limit and timeout must be positive and finite")
    return ModelConfig(role, provider, env.get(f"{prefix}_MODEL", "").strip(), base.rstrip("/"),
                       env.get(key_name, ""), limit, timeout)
