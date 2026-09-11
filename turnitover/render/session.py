"""ObservationSession: one headless Chromium page running the harness.

Statelessness: a session holds only the currently loaded program. Workers open one
session, load programs one after another, and dispose between them.
"""
from __future__ import annotations

import json
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from turnitover.checkers.base import PartMesh
from turnitover.core.actions import Detent, RuntimeProperty, detent_value
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing
from turnitover.render import protocol as P
from turnitover.render.transpile import CompileError, transpile_ts
from turnitover.render.views import ViewDef

log = logging.getLogger("turnitover.render")

HARNESS_ORIGIN = "https://harness.local"
DEFAULT_ARGS = (
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--disable-dev-shm-usage",
)


@dataclass(frozen=True)
class RenderConfig:
    harness_dir: Path
    esbuild_bin: Path
    width: int = 512
    height: int = 512
    browser_args: tuple[str, ...] = DEFAULT_ARGS
    browser_restart_every: int = 200


@dataclass(frozen=True)
class LoadInfo:
    parts: tuple[str, ...]
    joints: dict[str, dict]
    framing: Framing


@dataclass(frozen=True)
class ViewResult:
    view_id: str
    png: bytes
    render_ms: float
    camera: dict


@dataclass(frozen=True)
class ActuateResult:
    joint_id: str
    value: float
    png: bytes
    render_ms: float


class HarnessError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message


class ObservationSession:
    def __init__(self, cfg: RenderConfig, views: tuple[ViewDef, ...]):
        self.cfg = cfg
        self.views = views
        self._pw = None
        self._browser = None
        self._page = None
        self._loads = 0
        self._info: LoadInfo | None = None

    # ---- lifecycle -------------------------------------------------------
    def __enter__(self) -> "ObservationSession":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True, args=list(self.cfg.browser_args))
        self._page = self._browser.new_page(viewport={"width": self.cfg.width, "height": self.cfg.height})
        self._page.route(f"{HARNESS_ORIGIN}/**", self._serve_dist)
        self._page.goto(f"{HARNESS_ORIGIN}/index.html")
        self._page.wait_for_function("window.harnessReady === true")
        self._loads = 0

    def close(self) -> None:
        for obj in (self._browser, self._pw):
            try:
                if obj is not None:
                    obj.close() if obj is self._browser else obj.stop()
            except Exception:  # pragma: no cover - best effort
                pass
        self._browser = self._pw = self._page = None

    def restart(self) -> None:
        log.info("browser.restart", extra={"loads": self._loads})
        self.close()
        self.start()

    def gl_info(self) -> dict:
        return self._call("gl_info")

    def browser_version(self) -> str:
        return self._browser.version if self._browser else ""

    # ---- harness calls ---------------------------------------------------
    def load(self, program: ObjectProgram, framing: Framing | None) -> LoadInfo:
        if self.cfg.browser_restart_every and self._loads and self._loads % self.cfg.browser_restart_every == 0:
            self.restart()
        source_js = transpile_ts(program.source, self.cfg.esbuild_bin)
        req = {
            "source_js": source_js,
            "framing": "auto" if framing is None else {"center": list(framing.center), "radius": framing.radius},
            "width": self.cfg.width,
            "height": self.cfg.height,
            "views": [v.to_json() for v in self.views],
        }
        res: P.LoadResponse = self._call("load", req)
        self._loads += 1
        fr = res["framing"]
        self._info = LoadInfo(
            parts=tuple(res["parts"]),
            joints={j["id"]: j for j in res["joints"]},
            framing=Framing(center=tuple(float(x) for x in fr["center"]), radius=float(fr["radius"])),
        )
        return self._info

    @property
    def info(self) -> LoadInfo:
        if self._info is None:
            raise RuntimeError("no program loaded")
        return self._info

    def request_view(self, view_id: str) -> ViewResult:
        res = self._call("requestView", {"view_id": view_id})
        return ViewResult(view_id=view_id, png=P.decode_png(res["image_png_b64"]), render_ms=res["render_ms"], camera=res["camera"])

    def actuate_joint(self, joint_id: str, value: float) -> ActuateResult:
        res = self._call("actuateJoint", {"joint_id": joint_id, "value": value})
        return ActuateResult(joint_id=joint_id, value=res["value"], png=P.decode_png(res["image_png_b64"]), render_ms=res["render_ms"])

    def actuate(self, joint_id: str, detent: Detent) -> ActuateResult:
        limits = tuple(self.info.joints[joint_id]["limits"])
        return self.actuate_joint(joint_id, detent_value(detent, limits))

    def set_joints(self, values: dict[str, float]) -> None:
        self._call("setJoints", values)

    def reset_joints(self) -> None:
        self.set_joints({j: 0.0 for j in self.info.joints})

    def query_runtime(self, prop: RuntimeProperty) -> dict:
        return self._call("queryRuntime", {"property": prop.value})

    def export_geometry(self) -> dict[str, PartMesh]:
        res: P.GeometryExportDict = self._call("exportGeometry")
        return {
            p["name"]: PartMesh(vertices=P.decode_positions(p["positions_b64"]), faces=P.decode_indices(p["indices_b64"]))
            for p in res["parts"]
        }

    def dispose(self) -> None:
        if self._page is not None:
            self._call("dispose")
        self._info = None

    # ---- internals -------------------------------------------------------
    def _call(self, method: str, arg: Any = None) -> Any:
        from playwright.sync_api import Error as PlaywrightError

        js = f"(arg) => window.harness.{method}(arg)"
        try:
            return self._page.evaluate(js, arg)
        except PlaywrightError as e:
            raise _to_harness_error(str(e)) from None

    def _serve_dist(self, route, request) -> None:
        rel = request.url[len(HARNESS_ORIGIN) + 1:].split("?")[0]
        path = self.cfg.harness_dir / rel
        if not path.is_file():
            route.fulfill(status=404, body=f"not found: {rel}")
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "text/javascript"
        route.fulfill(status=200, body=path.read_bytes(), content_type=ctype)


def _to_harness_error(text: str) -> HarnessError:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            d = json.loads(text[start:end + 1])
            return HarnessError(d.get("stage", "runtime"), d.get("message", text))
        except json.JSONDecodeError:
            pass
    return HarnessError("runtime", text)


__all__ = ["ObservationSession", "RenderConfig", "LoadInfo", "ViewResult", "ActuateResult", "HarnessError", "CompileError"]
