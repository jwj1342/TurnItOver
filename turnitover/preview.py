"""Export inspectable screenshots and deterministic, frame-stepped demo videos.

Presentation only: orbit views and continuous motion do not change the judge action space.
"""
from __future__ import annotations

import dataclasses
import html
import io
import json
import logging
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from turnitover.core.actions import RuntimeProperty, ViewId
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.media import write_video
from turnitover.render.session import ObservationSession, RenderConfig
from turnitover.render.views import ViewDef
from turnitover.telemetry import git_state, now_iso

log = logging.getLogger("turnitover.preview")


def export_preview(program: ObjectProgram, render: RenderConfig, views: tuple[ViewDef, ...],
                   output: Path, *, fps: int = 24, seconds: float = 4.0,
                   videos: bool = True, ffmpeg: str = "ffmpeg") -> Path:
    """Export one program to a new/empty directory. seconds is per video motion segment."""
    if fps <= 0 or not math.isfinite(seconds) or seconds <= 0 or round(fps * seconds) < 2:
        raise ValueError("fps and seconds must produce at least two frames")
    if render.width <= 0 or render.height <= 0 or not views:
        raise ValueError("positive image dimensions and at least one view are required")
    encoder = shutil.which(ffmpeg) if videos else None
    if videos and encoder is None:
        raise ValueError("FFmpeg not found; use --ffmpeg PATH or --no-video for screenshots only")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new --out directory")
    output.mkdir(parents=True, exist_ok=True)
    shots = output / "screenshots"
    shots.mkdir(exist_ok=True)
    (output / "program.ts").write_text(program.source, encoding="utf-8")
    n_frames = round(fps * seconds)
    # Separate names avoid collisions with the configured research views.
    orbit = tuple(ViewDef(ViewId(f"__preview_orbit_{i}"), 360 * i / n_frames, 20,
                          "perspective", 1.2) for i in range(n_frames))
    motion_view = ViewDef(ViewId("__preview_motion"), 35, 20, "perspective", 1.25)
    all_views = views + orbit + (motion_view,)
    if len({v.id for v in all_views}) != len(all_views):
        raise ValueError("View IDs collide with reserved preview view IDs")
    screenshots: list[dict] = []
    movies: list[dict] = []
    manifest = {"status": "running", "created_at": now_iso(), "program_sha": program.sha,
                "fps": fps, "seconds_per_segment": seconds, "width": render.width,
                "height": render.height, "views": [v.to_json() for v in all_views],
                "git": git_state(Path(__file__).resolve().parents[1])}
    manifest_path = output / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_manifest()
    try:
        with ObservationSession(render, all_views) as session:
            info = session.load(program, framing=None)
            # Extra presentation margin accommodates parts that move outside the rest AABB.
            framing = dataclasses.replace(info.framing, radius=info.framing.radius * 1.25)
            info = session.load(program, framing=framing)
            manifest.update(framing=to_dict(framing), joints=info.joints,
                            browser=session.browser_version(), gl=session.gl_info())
            for i, view in enumerate(views):
                log.info("preview.screenshot %s", view.id)
                path = f"screenshots/{i:02d}.png"
                (output / path).write_bytes(session.request_view(view.id).png)
                screenshots.append({"title": str(view.id), "path": path, "joint_state": "rest"})
            for i, (jid, joint) in enumerate(info.joints.items()):
                session.reset_joints()
                session.request_view(motion_view.id)
                path = f"screenshots/joint-{i:02d}.png"
                (output / path).write_bytes(session.actuate_joint(jid, joint["limits"][1]).png)
                screenshots.append({"title": f"{jid}: upper limit", "path": path,
                                    "joint_id": jid, "value": joint["limits"][1]})
            session.reset_joints()
            session.request_view(views[0].id)
            manifest["runtime_stats"] = session.query_runtime(RuntimeProperty.STATS)
            _contact_sheet(output, screenshots)
            if videos:
                log.info("preview.video turntable")

                def orbit_frames():
                    session.reset_joints()
                    for view in orbit:
                        yield session.request_view(view.id).png

                count = write_video(output / "turntable.mp4", orbit_frames(), fps, encoder)
                movies.append({"title": "360 degree turntable", "path": "turntable.mp4",
                               "frames": count, "duration_seconds": count / fps})
                if info.joints:
                    log.info("preview.video joints")

                    def joint_frames():
                        session.request_view(motion_view.id)
                        for jid, joint in info.joints.items():
                            session.reset_joints()
                            lo, hi = joint["limits"]
                            for i in range(n_frames):
                                # Smooth lower -> upper -> lower sweep; both endpoints included.
                                u = 0.5 - 0.5 * math.cos(2 * math.pi * i / (n_frames - 1))
                                png = session.actuate_joint(jid, lo + (hi - lo) * u).png
                                yield _caption(png, f"{jid}  |  {lo + (hi - lo) * u:.3f}")

                    count = write_video(output / "joints.mp4", joint_frames(), fps, encoder)
                    movies.append({"title": "Joint motion (one joint at a time)", "path": "joints.mp4",
                                   "frames": count, "duration_seconds": count / fps,
                                   "joint_order": list(info.joints)})
            manifest.update(screenshots=screenshots, videos=movies)
        _gallery(output, screenshots, movies)
        manifest.update(status="complete", finished_at=now_iso())
        save_manifest()
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), screenshots=screenshots, videos=movies)
        save_manifest()
        raise
    return output / "index.html"


def _caption(png: bytes, title: str) -> bytes:
    with Image.open(io.BytesIO(png)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 30), fill="#16161a")
    draw.text((10, 8), title, fill="white")
    result = io.BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


def _contact_sheet(output: Path, screenshots: list[dict]) -> None:
    width, height, columns = 256, 286, 4
    sheet = Image.new("RGB", (columns * width, math.ceil(len(screenshots) / columns) * height), "#202024")
    draw = ImageDraw.Draw(sheet)
    for i, shot in enumerate(screenshots):
        x, y = (i % columns) * width, (i // columns) * height
        with Image.open(output / shot["path"]) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((width, 256))
        sheet.paste(thumb, (x + (width - thumb.width) // 2, y))
        draw.text((x + 10, y + 262), shot["title"], fill="white")
    sheet.save(output / "overview.png")


def _gallery(output: Path, screenshots: list[dict], movies: list[dict]) -> None:
    cards = "\n".join(f'<figure><a href="{s["path"]}"><img loading="lazy" src="{s["path"]}" '
                      f'alt="{html.escape(s["title"], quote=True)}"></a><figcaption>{html.escape(s["title"])}</figcaption></figure>'
                      for s in screenshots)
    clips = "\n".join(f'<figure><video controls loop muted playsinline preload="metadata" src="{m["path"]}"></video>'
                      f'<figcaption>{html.escape(m["title"])}</figcaption></figure>' for m in movies)
    page = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TurnItOver — Preview</title><style>
body{{margin:32px auto;padding:0 24px;max-width:1200px;background:#16161a;color:#eee;font:16px system-ui}}
a{{color:#9fc5ff}}h1{{margin-bottom:8px}}p{{color:#bdbdc9}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
figure{{margin:0;background:#202024;border-radius:12px;overflow:hidden}}img,video{{width:100%;display:block}}figcaption{{padding:12px}}
</style><h1>TurnItOver</h1><p>Rendered views and articulated motion. Videos use a fixed simulation step, independent of rendering speed.</p>
<p><a href="overview.png">Screenshot overview</a> · <a href="manifest.json">Run metadata</a> · <a href="program.ts">Program source</a></p>
{'<h2>Videos</h2><div class="grid">' + clips + '</div>' if movies else ''}
<h2>Screenshots</h2><div class="grid">{cards}</div></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
