"""Smoke test: launch headless Chromium through the harness, report the WebGL renderer, render one frame."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turnitover.assets.toy import toy_cabinet  # noqa: E402
from turnitover.core.program import ObjectProgram  # noqa: E402
from turnitover.render.session import ObservationSession, RenderConfig  # noqa: E402
from turnitover.render.views import load_views  # noqa: E402


def main() -> int:
    cfg = RenderConfig(harness_dir=ROOT / "web/dist", esbuild_bin=ROOT / "web/node_modules/.bin/esbuild", width=256, height=256)
    with ObservationSession(cfg, load_views(ROOT / "configs/views.yaml")) as s:
        print("browser:", s.browser_version())
        print("gl:", s.gl_info())
        info = s.load(ObjectProgram.from_spec(toy_cabinet()), framing=None)
        print("parts:", len(info.parts), "joints:", list(info.joints), "framing:", info.framing)
        v = s.request_view("front")
        print(f"render_ok={len(v.png) > 0} png_bytes={len(v.png)} render_ms={v.render_ms:.1f}")
        s.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
