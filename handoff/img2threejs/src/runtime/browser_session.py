from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .actions import ViewAction, pointer_motion


DEFAULT_BROWSER_ARGS = (
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--disable-dev-shm-usage",
)


@dataclass(frozen=True)
class BrowserConfig:
    width: int = 640
    height: int = 640
    headless: bool = True
    settle_ms: int = 350
    canvas_timeout_ms: int = 5000
    browser_args: tuple[str, ...] = DEFAULT_BROWSER_ARGS
    executable_path: str | None = None


@dataclass
class LoadResult:
    success: bool
    page_loaded: bool
    canvas_count: int
    canvas_box: dict[str, float] | None
    load_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    console_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionResult:
    action: str
    screenshot_path: str | None
    canvas_box: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BrowserRuntimeError(RuntimeError):
    pass


class BrowserSession:
    """Playwright-backed observation session for model-generated Three.js HTML.

    Generated HTML is treated as a black box. Camera actions are real pointer
    gestures on the largest visible canvas, so existing OrbitControls-style
    pages work without requiring a program ABI.
    """

    def __init__(self, config: BrowserConfig | None = None):
        self.config = config or BrowserConfig()
        self._pw = None
        self._browser = None
        self._page = None
        self._page_errors: list[str] = []
        self._console_errors: list[str] = []

    def __enter__(self) -> "BrowserSession":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        launch_kwargs: dict[str, Any] = {
            "headless": self.config.headless,
            "args": list(self.config.browser_args),
        }
        if self.config.executable_path:
            launch_kwargs["executable_path"] = self.config.executable_path
        self._browser = self._pw.chromium.launch(**launch_kwargs)
        self._page = self._browser.new_page(
            viewport={"width": self.config.width, "height": self.config.height}
        )
        self._page.on("pageerror", lambda error: self._page_errors.append(str(error)))
        self._page.on("console", self._on_console)

    def close(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._pw is not None:
                self._pw.stop()
            self._page = None
            self._browser = None
            self._pw = None

    def load_html(self, source: str) -> LoadResult:
        """Load HTML and report browser/runtime state without judging visual quality.

        Missing canvases are not recorded as JavaScript/page errors. Milestone 2
        performs that deterministic validity check separately in render_check.
        """
        self._ensure_started()
        assert self._page is not None
        self._page_errors.clear()
        self._console_errors.clear()
        load_errors: list[str] = []
        page_loaded = False

        try:
            # set_content alone preserves the JavaScript global environment.
            # Start a fresh document for retries and verifier reloads.
            self._page.goto("about:blank", wait_until="load")
            self._page.set_content(source, wait_until="load")
            page_loaded = True
            self._page.wait_for_timeout(self.config.settle_ms)
        except Exception as exc:
            load_errors.append(str(exc))

        if page_loaded:
            try:
                self._page.wait_for_selector(
                    "canvas",
                    state="visible",
                    timeout=self.config.canvas_timeout_ms,
                )
            except Exception:
                # No visible canvas is a render-validity outcome, not a page error.
                pass

        canvases = self._visible_canvases() if page_loaded else []
        box = canvases[0]["box"] if canvases else None
        success = (
            page_loaded
            and bool(canvases)
            and not load_errors
            and not self._page_errors
        )
        return LoadResult(
            success=success,
            page_loaded=page_loaded,
            canvas_count=len(canvases),
            canvas_box=box,
            load_errors=load_errors,
            page_errors=list(self._page_errors),
            console_errors=list(self._console_errors),
        )

    def capture(self, path: str | Path | None = None) -> bytes:
        locator, _box = self._largest_canvas()
        png = locator.screenshot(type="png")
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(png)
        return png

    def perform(
        self,
        action: ViewAction,
        *,
        screenshot_path: str | Path | None = None,
    ) -> ActionResult:
        locator, box = self._largest_canvas()
        page = self._page
        assert page is not None

        center_x = box["x"] + box["width"] / 2.0
        center_y = box["y"] + box["height"] / 2.0
        motion = pointer_motion(action, width=box["width"], height=box["height"])

        if motion.wheel_y:
            locator.hover()
            page.mouse.wheel(0, motion.wheel_y)
        elif motion.dx or motion.dy:
            page.mouse.move(center_x, center_y)
            page.mouse.down()
            page.mouse.move(center_x + motion.dx, center_y + motion.dy, steps=8)
            page.mouse.up()

        page.wait_for_timeout(self.config.settle_ms)

        saved: str | None = None
        if screenshot_path is not None:
            target = Path(screenshot_path)
            self.capture(target)
            saved = str(target)

        return ActionResult(action=action.value, screenshot_path=saved, canvas_box=box)

    @property
    def page_errors(self) -> tuple[str, ...]:
        return tuple(self._page_errors)

    @property
    def console_errors(self) -> tuple[str, ...]:
        return tuple(self._console_errors)

    def _on_console(self, message: Any) -> None:
        if getattr(message, "type", None) == "error":
            self._console_errors.append(message.text)

    def _ensure_started(self) -> None:
        if self._page is None:
            raise BrowserRuntimeError(
                "BrowserSession is not started; use it as a context manager or call start()"
            )

    def _visible_canvases(self) -> list[dict[str, Any]]:
        self._ensure_started()
        assert self._page is not None
        items = self._page.locator("canvas").evaluate_all(
            """
            (els) => els.map((el, index) => {
              const rect = el.getBoundingClientRect();
              const style = getComputedStyle(el);
              const visible = rect.width > 1 && rect.height > 1 &&
                style.display !== 'none' && style.visibility !== 'hidden';
              return {
                index,
                visible,
                area: rect.width * rect.height,
                box: {x: rect.x, y: rect.y, width: rect.width, height: rect.height}
              };
            }).filter(x => x.visible).sort((a, b) => b.area - a.area)
            """
        )
        return items

    def _largest_canvas(self):
        canvases = self._visible_canvases()
        if not canvases:
            raise BrowserRuntimeError("no visible canvas found in the loaded HTML")
        assert self._page is not None
        entry = canvases[0]
        return self._page.locator("canvas").nth(entry["index"]), entry["box"]
