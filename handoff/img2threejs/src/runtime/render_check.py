from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from .browser_session import BrowserSession, LoadResult


@dataclass
class RenderCheckResult:
    success: bool
    page_loaded: bool
    canvas_found: bool
    canvas_nonempty: bool
    load_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    console_errors: list[str] = field(default_factory=list)
    screenshot_path: str | None = None
    image_stddev: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def repair_message(self) -> str:
        reasons: list[str] = []
        if not self.page_loaded:
            reasons.append("page failed to load")
        if not self.canvas_found:
            reasons.append("no visible canvas was found")
        if self.canvas_found and not self.canvas_nonempty:
            reasons.append("the rendered canvas appears blank or uniform")
        if self.load_errors:
            reasons.append("load errors: " + " | ".join(self.load_errors))
        if self.page_errors:
            reasons.append("page errors: " + " | ".join(self.page_errors))
        if self.console_errors:
            reasons.append("console errors: " + " | ".join(self.console_errors))
        return "; ".join(reasons) if reasons else "render check passed"


def _image_stddev(png: bytes) -> float:
    with Image.open(BytesIO(png)).convert("RGB") as image:
        stat = ImageStat.Stat(image)
        return float(sum(stat.stddev) / len(stat.stddev))


def check_render(
    session: BrowserSession,
    load: LoadResult,
    output_dir: str | Path,
    *,
    blank_stddev_threshold: float = 1.0,
) -> RenderCheckResult:
    """Deterministically decide whether the current HTML produced useful pixels.

    The check intentionally does not use a VLM. A render passes when the page
    loaded, a visible canvas exists, there are no page/runtime errors, and the
    largest canvas screenshot has non-trivial pixel variation.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    screenshot_path: str | None = None
    stddev: float | None = None
    canvas_nonempty = False

    if load.canvas_count > 0:
        target = out / "render.png"
        png = session.capture(target)
        screenshot_path = str(target)
        stddev = _image_stddev(png)
        canvas_nonempty = stddev >= blank_stddev_threshold

    fatal_console = bool(load.console_errors)
    success = (
        load.page_loaded
        and load.canvas_count > 0
        and canvas_nonempty
        and not load.load_errors
        and not load.page_errors
        and not fatal_console
    )

    return RenderCheckResult(
        success=success,
        page_loaded=load.page_loaded,
        canvas_found=load.canvas_count > 0,
        canvas_nonempty=canvas_nonempty,
        load_errors=list(load.load_errors),
        page_errors=list(load.page_errors),
        console_errors=list(load.console_errors),
        screenshot_path=screenshot_path,
        image_stddev=stddev,
    )
