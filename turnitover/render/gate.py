"""Deterministic renderability gate built on the shared observation harness."""
from __future__ import annotations

import dataclasses
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Error as PlaywrightError

from turnitover.core.program import ObjectProgram
from turnitover.render.session import CompileError, HarnessError, ObservationSession, RenderConfig
from turnitover.render.views import ViewDef


@dataclass(frozen=True)
class RenderGateResult:
    success: bool
    stage: str
    error_type: str | None = None
    message: str | None = None
    image_ref: str | None = None
    image_stddev: float | None = None
    failure_kind: str | None = None


def image_stddev(png: bytes) -> float:
    with Image.open(io.BytesIO(png)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return float(rgb.std())


def run_render_gate(program: ObjectProgram, output: Path, render: RenderConfig,
                    views: tuple[ViewDef, ...], blank_stddev_threshold: float = 1.0) -> RenderGateResult:
    if not views:
        raise ValueError("At least one configured view is required")
    if blank_stddev_threshold < 0:
        raise ValueError("Blank-image threshold must be nonnegative")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Render-gate output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidate.ts").write_text(program.source, encoding="utf-8")
    stage = "browser_start"
    try:
        with ObservationSession(render, views) as session:
            stage = "program_load"
            session.load(program, framing=None)
            stage = "request_view"
            view = next((item for item in views if str(item.id) == "front"), views[0])
            observation = session.request_view(view.id)
            (output / "render.png").write_bytes(observation.png)
            deviation = image_stddev(observation.png)
            if deviation <= blank_stddev_threshold:
                result = RenderGateResult(False, "pixel_check", "BlankRender",
                                          "Rendered image is blank or nearly constant.", "render.png", deviation,
                                          failure_kind="candidate")
            else:
                result = RenderGateResult(True, "complete", image_ref="render.png", image_stddev=deviation)
    except CompileError as exc:
        result = RenderGateResult(False, "compile", type(exc).__name__, str(exc),
                                  failure_kind="candidate")
    except HarnessError as exc:
        result = RenderGateResult(False, exc.stage or stage, type(exc).__name__, exc.message,
                                  failure_kind="candidate")
    except (PlaywrightError, OSError) as exc:
        result = RenderGateResult(False, stage, type(exc).__name__,
                                  "Browser environment could not complete the render gate.",
                                  failure_kind="environment")
    _write_json(output / "result.json", dataclasses.asdict(result))
    return result


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
