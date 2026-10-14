"""Replaceable teacher backends with no hidden retry or model fallback."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol, Sequence

from turnitover.models.client import ModelResult, complete
from turnitover.models.config import ModelConfig
from turnitover.models.local_qwen import LocalQwen


KIMI_MODELS = frozenset({"k3", "k3-256k", "kimi-for-coding"})


class Teacher(Protocol):
    def __call__(self, prompt: str, images: list[Path]) -> ModelResult: ...
    def public(self) -> dict: ...


class MockTeacher:
    def __init__(self, decisions: Sequence[dict | str]):
        self.decisions = iter(decisions)

    def __call__(self, prompt: str, images: list[Path]) -> ModelResult:
        value = next(self.decisions)
        text = value if isinstance(value, str) else json.dumps(value)
        return ModelResult(text, "deterministic-mock", {"input_tokens": 0, "output_tokens": 0},
                           "completed", 0.0)

    def public(self) -> dict:
        return {"kind": "mock", "model": "deterministic-mock", "paid": False}


class LocalQwenTeacher:
    def __init__(self, model_path: Path, max_tokens: int = 512):
        self.model_path = model_path.resolve()
        self.call = LocalQwen(self.model_path, max_tokens=max_tokens)

    def __call__(self, prompt: str, images: list[Path]) -> ModelResult:
        return self.call(prompt, images)

    def public(self) -> dict:
        return {"kind": "local_qwen", "model": self.model_path.name, "paid": False,
                "local_files_only": True, "max_tokens": self.call.max_tokens}


@dataclass(frozen=True)
class KimiPricing:
    input_rmb_per_million: float
    output_rmb_per_million: float
    verified_at: str
    source: str

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        values = (input_tokens, output_tokens, self.input_rmb_per_million, self.output_rmb_per_million)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Token and price estimates must be nonnegative")
        return (input_tokens * self.input_rmb_per_million
                + output_tokens * self.output_rmb_per_million) / 1_000_000


class KimiTeacher:
    """Bounded Kimi adapter; callers enforce the episode call cap."""

    def __init__(self, config: ModelConfig, pricing: KimiPricing | None, *, transmission_reviewed: bool,
                 estimated_input_tokens: int, max_cost_rmb: float = 1.0,
                 max_real_requests: int = 1):
        if config.model not in KIMI_MODELS:
            raise ValueError("Kimi teacher model is not in the Issue #4 tested candidate set")
        config.validate()
        if not transmission_reviewed:
            raise ValueError("Third-party asset transmission terms were not reviewed")
        if pricing is not None and (pricing.verified_at != date.today().isoformat() or not pricing.source):
            raise ValueError("Kimi pricing must be verified today from an identified source")
        if type(max_real_requests) is not int or max_real_requests < 1:
            raise ValueError("Kimi real-request cap must be a positive integer")
        estimate = pricing.estimate(estimated_input_tokens, config.max_tokens) if pricing else None
        if estimate is not None and estimate > max_cost_rmb:
            raise ValueError("Estimated Kimi request cost exceeds the configured cap")
        self.config, self.pricing = config, pricing
        self.estimated_input_tokens = estimated_input_tokens
        self.estimated_cost_rmb = estimate
        self.max_cost_rmb = max_cost_rmb
        self.max_real_requests = max_real_requests
        self.calls = 0

    def __call__(self, prompt: str, images: list[Path]) -> ModelResult:
        if self.calls >= self.max_real_requests:
            raise RuntimeError("Kimi real-request cap reached")
        self.calls += 1
        return complete(self.config, prompt, images)

    def public(self) -> dict:
        return {"kind": "kimi", "model": self.config.model, "paid": True,
                "max_real_requests": self.max_real_requests,
                "estimated_input_tokens": self.estimated_input_tokens,
                "estimated_max_cost_rmb": self.estimated_cost_rmb,
                "cost_cap_rmb": self.max_cost_rmb if self.pricing else None,
                "pricing": self.pricing.__dict__ if self.pricing else None,
                "billing_basis": "token_pricing" if self.pricing else "unknown_gateway_or_membership_quota"}
