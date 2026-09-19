from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.agents import CodeAgent
from src.config import Settings
from src.pipeline.retry import run_with_render_retry
from src.runtime import BrowserConfig, BrowserSession
from src.trajectory import CodegenTrajectoryWriter


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenHands/Qwen image-to-Three.js generation with render retry.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--chromium", default=None, help="Optional system Chromium executable path")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.validate_model()

    run_id = args.run_id or time.strftime("codegen-%Y%m%d-%H%M%S")
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    writer = CodegenTrajectoryWriter(run_dir, args.reference)
    agent = CodeAgent(settings)
    current_html = ""
    call_index = 0

    def generate_or_repair(previous, attempt_index: int) -> str:
        nonlocal current_html, call_index
        workspace = run_dir / "workspaces" / f"attempt_{attempt_index:02d}"
        if previous is None:
            result = agent.generate(args.reference, workspace)
            kind = "generate"
        else:
            result = agent.repair_runtime(args.reference, current_html, previous, workspace)
            kind = "repair"
        current_html = result.html
        writer.record_call(call_index, kind, result)
        call_index += 1
        return current_html

    browser_config = BrowserConfig(
        headless=not args.headed,
        executable_path=args.chromium,
    )
    with BrowserSession(browser_config) as session:
        result = run_with_render_retry(
            session,
            generate_or_repair,
            run_dir / "render_attempts",
            max_attempts=1 + min(3, max(0, settings.max_render_retries)),
        )

    trajectory = writer.finalize(
        success=result.success,
        attempts=len(result.attempts),
        final_html=result.final_html,
    )
    summary = {
        "run_dir": str(run_dir),
        "trajectory": str(trajectory),
        "success": result.success,
        "attempts": len(result.attempts),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
