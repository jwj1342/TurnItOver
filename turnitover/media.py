"""Encode rendered PNG frames without keeping a video in memory."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


def write_video(path: Path, frames: Iterable[bytes], fps: int, ffmpeg: str) -> int:
    """Write an H.264 MP4 atomically; return the number of encoded frames."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    count = 0
    with tempfile.TemporaryDirectory(prefix=".encode-", dir=path.parent) as tmp:
        target = Path(tmp) / "video.mp4"
        with tempfile.TemporaryFile() as errors:
            proc = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "image2pipe", "-framerate", str(fps), "-vcodec", "png", "-i", "pipe:0",
                 "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(target)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors,
            )
            try:
                try:
                    for frame in frames:
                        proc.stdin.write(frame)
                        count += 1
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                code = proc.wait(timeout=120)
                if code or not count:
                    errors.seek(0)
                    raise RuntimeError(f"Video encoding failed ({code}): {errors.read().decode(errors='replace')}")
                target.replace(path)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if not proc.stdin.closed:
                    proc.stdin.close()
    return count
