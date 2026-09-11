"""ObjectProgram: TypeScript source implementing the Program ABI, plus optional spec."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from turnitover.core.spec import AssetSpec


@dataclass(frozen=True)
class ObjectProgram:
    source: str
    spec: AssetSpec | None = None
    language: Literal["ts"] = "ts"

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def from_spec(spec: AssetSpec) -> "ObjectProgram":
        from turnitover.assets.emit import emit_program

        return ObjectProgram(source=emit_program(spec), spec=spec)
