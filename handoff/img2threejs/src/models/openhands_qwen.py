from __future__ import annotations

from src.config import Settings


def build_qwen_llm(settings: Settings, *, usage_id: str = "code-agent"):
    """Build an OpenHands LLM configured for an OpenAI-compatible Qwen endpoint.

    OpenHands imports are intentionally local so the Streamlit viewer and unit
    tests can run without installing the heavier agent dependencies.
    """
    settings.validate_model()

    try:
        from openhands.sdk import LLM
        from pydantic import SecretStr
    except ImportError as exc:  # pragma: no cover - exercised only in agent env
        raise RuntimeError(
            "OpenHands agent dependencies are missing. Install requirements-agent.txt."
        ) from exc

    kwargs = {
        "usage_id": usage_id,
        "model": settings.qwen_model,
        "base_url": settings.qwen_base_url,
        "api_key": SecretStr(settings.qwen_api_key),
        "max_output_tokens": settings.qwen_max_output_tokens,
        # Custom OpenAI-compatible model names may be absent from LiteLLM's
        # model registry. This agent always sends a reference image, so declare
        # the endpoint's required vision capability explicitly.
        "capability_overrides": {"supports_vision": True},
    }
    if settings.qwen_temperature is not None:
        kwargs["temperature"] = settings.qwen_temperature

    return LLM(**kwargs)
