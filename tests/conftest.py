from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config, items):
    dist_ok = (ROOT / "web/dist/harness.js").is_file() and (ROOT / "web/node_modules/.bin/esbuild").exists()
    browsers = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ROOT / ".cache/ms-playwright"))
    browser_ok = browsers.is_dir() and any(browsers.iterdir())
    reason = None
    if not dist_ok:
        reason = "web/dist not built (run: cd web && npm ci && npm run build)"
    elif not browser_ok:
        reason = "no Playwright browser (run: python -m playwright install chromium-headless-shell)"
    if reason:
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if "browser" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def toy_spec():
    from turnitover.assets.toy import toy_cabinet

    return toy_cabinet()


@pytest.fixture
def toy_program(toy_spec):
    from turnitover.core.program import ObjectProgram

    return ObjectProgram.from_spec(toy_spec)


@pytest.fixture
def views():
    from turnitover.render.views import load_views

    return load_views(ROOT / "configs/views.yaml")


@pytest.fixture
def render_config():
    from turnitover.render.session import RenderConfig

    return RenderConfig(harness_dir=ROOT / "web/dist", esbuild_bin=ROOT / "web/node_modules/.bin/esbuild", width=256, height=256)


@pytest.fixture(scope="module")
def session_factory(request):
    """Module-scoped browser session to keep browser tests fast."""
    from turnitover.render.session import ObservationSession, RenderConfig
    from turnitover.render.views import load_views

    cfg = RenderConfig(harness_dir=ROOT / "web/dist", esbuild_bin=ROOT / "web/node_modules/.bin/esbuild", width=256, height=256)
    session = ObservationSession(cfg, load_views(ROOT / "configs/views.yaml"))
    session.start()
    request.addfinalizer(session.close)
    return session


@pytest.fixture
def sample():
    from turnitover.core.actions import ActuateJoint, Detent, Observation, QueryRuntime, RequestView, RuntimeProperty, Stop, ViewId
    from turnitover.core.sample import SCHEMA_VERSION, CheckResult, CorruptionSpec, Framing, Invariant, Sample

    return Sample(
        schema_version=SCHEMA_VERSION, sample_id="toy-0", run_id="t", idx=0, seed=7, asset_id="toy", source_id="toy",
        program_sha="abc", program_source="export default 1", reference_sha="def",
        corruptions=(CorruptionSpec("structure.part_offset", ("door",), 0.5, {"m": 1.0}, {"part": "door"}),),
        framing=Framing((0.0, 0.4, 0.0), 0.6),
        trajectory=(
            Observation(0, RequestView(ViewId("front")), "obs/00.png", {"render_ms": 1.0}, 2.0),
            Observation(1, ActuateJoint("door", Detent.LIMIT), "obs/01.png", None, 2.0),
            Observation(2, QueryRuntime(RuntimeProperty.STATS), None, {"draw_calls": 3}, 1.0),
            Observation(3, Stop(), None, None, 0.0),
        ),
        checks=(CheckResult("runtime.budget", True, (Invariant("triangles", 12.0, 100.0, True),), 1.0),),
        compile_ok=True, error=None, timing_ms={"load": 3.0}, host="h", created_at="2026-09-11T00:00:00+00:00",
    )
