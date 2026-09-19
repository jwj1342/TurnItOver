from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.runtime.browser_session import BrowserSession
from src.runtime.render_check import RenderCheckResult, check_render


GenerateOrRepair = Callable[[RenderCheckResult | None, int], str]


@dataclass
class RenderAttempt:
    attempt: int
    html: str
    check: RenderCheckResult


@dataclass
class RenderAttemptResult:
    success: bool
    final_html: str
    attempts: list[RenderAttempt] = field(default_factory=list)


def run_with_render_retry(
    session: BrowserSession,
    generate_or_repair: GenerateOrRepair,
    output_dir: str | Path,
    *,
    max_attempts: int = 3,
) -> RenderAttemptResult:
    """Run deterministic render validation with bounded generation/repair retries.

    ``generate_or_repair`` receives the previous failed render check (or ``None``
    for the first attempt) and the zero-based attempt index, and returns complete
    HTML for the next candidate.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    attempts: list[RenderAttempt] = []
    previous: RenderCheckResult | None = None
    final_html = ""

    for attempt_index in range(max_attempts):
        final_html = generate_or_repair(previous, attempt_index)
        attempt_dir = out / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "candidate.html").write_text(final_html, encoding="utf-8")

        load = session.load_html(final_html)
        check = check_render(session, load, attempt_dir)
        (attempt_dir / "check.json").write_text(
            json.dumps(check.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        attempts.append(RenderAttempt(attempt=attempt_index, html=final_html, check=check))
        if check.success:
            return RenderAttemptResult(success=True, final_html=final_html, attempts=attempts)
        previous = check

    return RenderAttemptResult(success=False, final_html=final_html, attempts=attempts)
