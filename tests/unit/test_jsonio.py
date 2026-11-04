import json

import pytest

from turnitover.core.jsonio import write_json_atomic


def test_atomic_json_writer_uses_utf8_and_rejects_nan(tmp_path):
    path = tmp_path / "artifact.json"
    write_json_atomic(path, {"label": "柜体"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"label": "柜体"}
    assert "柜体" in path.read_text(encoding="utf-8")
    assert not path.with_suffix(".json.tmp").exists()

    with pytest.raises(ValueError):
        write_json_atomic(path, {"invalid": float("nan")})
