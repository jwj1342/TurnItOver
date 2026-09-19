from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class CodegenTrajectoryWriter:
    def __init__(self, run_dir: str | Path, reference_image: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        source = Path(reference_image)
        self.reference_path = self.run_dir / f"reference{source.suffix.lower() or '.png'}"
        if source.resolve() != self.reference_path.resolve():
            shutil.copy2(source, self.reference_path)
        self.calls: list[dict[str, Any]] = []

    def record_call(self, index: int, kind: str, result: Any) -> Path:
        call_dir = self.run_dir / f"call_{index:02d}_{kind}"
        call_dir.mkdir(parents=True, exist_ok=True)
        (call_dir / "code.html").write_text(result.html, encoding="utf-8")
        (call_dir / "reasoning.txt").write_text(result.reasoning or "", encoding="utf-8")
        payload = result.to_dict(include_html=False)
        payload["kind"] = kind
        (call_dir / "agent.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        self.calls.append({"index": index, "kind": kind, "dir": call_dir.name, "usage": result.usage})
        return call_dir

    def finalize(self, *, success: bool, attempts: int, final_html: str) -> Path:
        (self.run_dir / "final.html").write_text(final_html, encoding="utf-8")
        payload = {
            "schema_version": 1,
            "reference": self.reference_path.name,
            "success": success,
            "attempts": attempts,
            "calls": self.calls,
            "final_html": "final.html",
        }
        path = self.run_dir / "trajectory.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return path
