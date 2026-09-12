import io
import shutil
import subprocess

import pytest
from PIL import Image

from turnitover.media import write_video


def test_video_encodes_all_frames_and_pads_odd_dimensions(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("system FFmpeg not installed")
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
    if "libx264 " not in encoders.stdout:
        pytest.skip("FFmpeg has no libx264 encoder")

    def frames():
        for color in ("red", "green", "blue"):
            buf = io.BytesIO()
            Image.new("RGB", (31, 33), color).save(buf, format="PNG")
            yield buf.getvalue()

    path = tmp_path / "clip.mp4"
    assert write_video(path, frames(), 3, ffmpeg) == 3
    decoded = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True)
    assert len(decoded.stdout) == 3 * 32 * 34 * 3
    assert decoded.stdout[:32 * 34 * 3] != decoded.stdout[-32 * 34 * 3:]


def test_frame_error_does_not_publish_partial_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("system FFmpeg not installed")

    def frames():
        raise ValueError("render failed")
        yield b""

    path = tmp_path / "clip.mp4"
    with pytest.raises(ValueError, match="render failed"):
        write_video(path, frames(), 24, ffmpeg)
    assert not path.exists()
    assert not list(tmp_path.glob(".encode-*"))
