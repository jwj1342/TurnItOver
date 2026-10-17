from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

from src.agents import CodeAgent, VerifierAgent
from src.config import Settings
from src.pipeline import PipelineController
from src.runtime import BrowserConfig, BrowserSession
from src.verifier import load_action_space


def _resolve_chromium(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    from_env = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    if from_env:
        return from_env
    return shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Milestone 5 CodeAgent -> VerifierAgent iterative pipeline."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--chromium", default=None)
    parser.add_argument("--max-visual-revisions", type=int, default=None)
    parser.add_argument("--action-space", default=None)
    parser.add_argument("--action-space-config", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.max_visual_revisions is not None:
        settings = replace(settings, max_visual_revisions=args.max_visual_revisions)
    settings.validate_model()

    mode = args.action_space or settings.view_action_space
    action_space_config = args.action_space_config or settings.view_action_space_config
    action_space = load_action_space(mode, action_space_config)

    run_id = args.run_id or time.strftime("pipeline-%Y%m%d-%H%M%S")
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    reference_copy = run_dir / f"reference{args.reference.suffix.lower() or '.png'}"
    shutil.copy2(args.reference, reference_copy)

    browser_config = BrowserConfig(
        headless=not args.headed,
        executable_path=_resolve_chromium(args.chromium),
    )
    with BrowserSession(browser_config) as session:
        controller = PipelineController(
            settings=settings,
            code_agent=CodeAgent(settings),
            verifier_agent=VerifierAgent(settings),
            browser_session=session,
            action_space=action_space,
        )
        result = controller.run(reference_copy, run_dir)

    summary = {
        "run_dir": str(run_dir),
        "status": result.status,
        "accepted": result.accepted,
        "visual_revisions": result.visual_revisions,
        "rounds": len(result.rounds),
        "action_space": action_space.name,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result.status != "render_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
