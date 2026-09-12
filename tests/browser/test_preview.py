import dataclasses
import json

import pytest
from PIL import Image, ImageChops

from turnitover.preview import export_preview

pytestmark = pytest.mark.browser


def test_preview_exports_images_and_metadata(tmp_path, toy_program, render_config, views):
    output = tmp_path / "preview"
    page = export_preview(toy_program, dataclasses.replace(render_config, width=128, height=128),
                          views[:2], output, videos=False)
    assert page.is_file() and "video controls" not in page.read_text()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["screenshots"]) == 5 and manifest["videos"] == []
    with Image.open(output / "screenshots/00.png") as a, Image.open(output / "screenshots/01.png") as b:
        assert a.size == (128, 128)
        assert ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox() is not None
    assert (output / "overview.png").is_file()
    assert (output / "program.ts").read_text() == toy_program.source
    with pytest.raises(ValueError, match="not empty"):
        export_preview(toy_program, render_config, views, output, videos=False)
