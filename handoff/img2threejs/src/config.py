from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    qwen_base_url: str
    qwen_api_key: str
    qwen_model: str
    qwen_max_output_tokens: int = 8192
    qwen_temperature: float | None = None
    code_agent_max_steps: int = 8
    verifier_agent_max_steps: int = 10
    verifier_action_budget: int = 6
    view_action_space: str = "relative_discrete_v1"
    view_action_space_config: Path | None = None
    max_render_retries: int = 3  # Repairs after the initial render check (0..3).
    max_visual_revisions: int = 2
    runs_dir: Path = Path("data/runs")

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        load_dotenv(dotenv_path=env_file, override=False)

        base_url = os.getenv("QWEN_BASE_URL", "").strip()
        api_key = os.getenv("QWEN_API_KEY", "").strip()
        model = os.getenv("QWEN_MODEL", "").strip()
        max_tokens_raw = os.getenv("QWEN_MAX_OUTPUT_TOKENS", "8192").strip() or "8192"
        temperature_raw = os.getenv("QWEN_TEMPERATURE", "").strip()
        max_steps_raw = os.getenv("CODE_AGENT_MAX_STEPS", "8").strip() or "8"
        verifier_steps_raw = os.getenv("VERIFIER_AGENT_MAX_STEPS", "10").strip() or "10"
        verifier_budget_raw = os.getenv("VERIFIER_ACTION_BUDGET", "6").strip() or "6"
        view_action_space = os.getenv("VIEW_ACTION_SPACE", "relative_discrete_v1").strip() or "relative_discrete_v1"
        view_config_raw = os.getenv("VIEW_ACTION_SPACE_CONFIG", "").strip()
        max_retries_raw = os.getenv("MAX_RENDER_RETRIES", "3").strip() or "3"
        max_visual_revisions_raw = os.getenv("MAX_VISUAL_REVISIONS", "2").strip() or "2"
        runs_dir = Path(os.getenv("RUNS_DIR", "data/runs").strip() or "data/runs")

        return cls(
            qwen_base_url=base_url,
            qwen_api_key=api_key,
            qwen_model=model,
            qwen_max_output_tokens=int(max_tokens_raw),
            qwen_temperature=(float(temperature_raw) if temperature_raw else None),
            code_agent_max_steps=int(max_steps_raw),
            verifier_agent_max_steps=int(verifier_steps_raw),
            verifier_action_budget=int(verifier_budget_raw),
            view_action_space=view_action_space,
            view_action_space_config=(Path(view_config_raw) if view_config_raw else None),
            max_render_retries=int(max_retries_raw),
            max_visual_revisions=int(max_visual_revisions_raw),
            runs_dir=runs_dir,
        )

    def validate_model(self) -> None:
        missing = []
        if not self.qwen_base_url:
            missing.append("QWEN_BASE_URL")
        if not self.qwen_api_key:
            missing.append("QWEN_API_KEY")
        if not self.qwen_model:
            missing.append("QWEN_MODEL")
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"Missing model configuration: {joined}. Copy .env.example to .env and fill them in."
            )

        if self.code_agent_max_steps < 1:
            raise RuntimeError("CODE_AGENT_MAX_STEPS must be >= 1")
        if self.verifier_agent_max_steps < 1:
            raise RuntimeError("VERIFIER_AGENT_MAX_STEPS must be >= 1")
        if self.verifier_action_budget < 1:
            raise RuntimeError("VERIFIER_ACTION_BUDGET must be >= 1")
        if not 0 <= self.max_render_retries <= 3:
            raise RuntimeError("MAX_RENDER_RETRIES must be between 0 and 3")
        if self.max_visual_revisions < 0:
            raise RuntimeError("MAX_VISUAL_REVISIONS must be >= 0")
