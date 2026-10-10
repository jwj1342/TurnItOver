"""Small atomic JSON artifact writer shared by runtime entry points."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
