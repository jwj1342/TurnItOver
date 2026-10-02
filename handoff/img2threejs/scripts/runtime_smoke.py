from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runtime import BrowserConfig, BrowserSession, check_render, parse_action


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise render validation and active-view actions on a saved Three.js HTML file."
    )
    parser.add_argument("html", type=Path, help="Path to a complete HTML file produced by the model")
    parser.add_argument("--out", type=Path, default=Path("data/runs/runtime-smoke"))
    parser.add_argument(
        "--actions",
        nargs="*",
        default=["orbit_right", "zoom_in"],
        help="Sequence of orbit/zoom actions to execute after render validation",
    )
    parser.add_argument("--headed", action="store_true", help="Show Chromium instead of running headless")
    parser.add_argument(
        "--chromium",
        type=str,
        default=None,
        help="Optional Chromium executable path; otherwise Playwright's installed browser is used",
    )
    args = parser.parse_args()

    source = args.html.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)

    report: dict = {"html": str(args.html), "steps": []}
    config = BrowserConfig(headless=not args.headed, executable_path=args.chromium)
    with BrowserSession(config) as session:
        load = session.load_html(source)
        report["load"] = load.to_dict()
        render = check_render(session, load, args.out)
        report["render_check"] = render.to_dict()
        report["repair_message"] = render.repair_message()

        if not render.success:
            (args.out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 1

        for index, raw_action in enumerate(args.actions, start=1):
            action = parse_action(raw_action)
            screenshot = args.out / f"{index:02d}_{action.value}.png"
            result = session.perform(action, screenshot_path=screenshot)
            report["steps"].append(result.to_dict())

        report["page_errors"] = list(session.page_errors)
        report["console_errors"] = list(session.console_errors)

    (args.out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
