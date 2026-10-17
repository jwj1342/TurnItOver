from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from src.agents import VerifierAgent
from src.config import Settings
from src.runtime import BrowserConfig, BrowserSession
from src.verifier import (
    PoseBrowserRuntime,
    PoseGridActionSpace,
    RelativeBrowserRuntime,
    RelativeDiscreteActionSpace,
    VerifierEpisode,
    load_action_space,
)


def _resolve_chromium() -> str | None:
    explicit = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    if explicit:
        return explicit
    return shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the active OpenHands verifier on one Three.js HTML candidate.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("html", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/runs/verifier-smoke"))
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument(
        "--action-space",
        default=None,
        help="Override VIEW_ACTION_SPACE (relative_discrete_v1 or pose_grid_v2)",
    )
    parser.add_argument(
        "--action-space-config",
        type=Path,
        default=None,
        help="Override VIEW_ACTION_SPACE_CONFIG",
    )
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.validate_model()
    source = args.html.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)

    mode = args.action_space or settings.view_action_space
    config_path = args.action_space_config or settings.view_action_space_config
    action_space = load_action_space(mode, config_path)

    config = BrowserConfig(
        headless=not args.headed,
        executable_path=_resolve_chromium(),
    )
    with BrowserSession(config) as session:
        if isinstance(action_space, RelativeDiscreteActionSpace):
            runtime = RelativeBrowserRuntime(session, action_space)
        elif isinstance(action_space, PoseGridActionSpace):
            runtime = PoseBrowserRuntime(session, action_space)
        else:
            raise TypeError(f"unsupported action space type: {type(action_space).__name__}")

        episode = VerifierEpisode(
            runtime,
            source,
            args.out / "verifier",
            action_budget=args.budget or settings.verifier_action_budget,
            action_space=action_space,
        )
        result = VerifierAgent(settings).verify(args.reference, episode)

    payload = result.to_dict()
    payload["action_space"] = action_space.to_dict()
    (args.out / "verifier_agent.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps({
        "verdict": result.verdict,
        "feedback": result.feedback,
        "selected_view_ids": result.selected_view_ids,
        "observations": len(result.observations),
        "action_space": action_space.name,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
