"""One frozen API configuration and a shared, durable request-count limit.

The limit counts attempts, including failures; it is not a dollar spending cap.
No retry, model substitution, or price estimation occurs here.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from turnitover.models.client import ModelError, ModelResult, complete
from turnitover.models.config import ModelConfig


def _write(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class FrozenAPI:
    """Serialize Generator/Verifier calls against the same immutable configuration.

    Use a fresh output directory for each run. Completed and interrupted ledgers
    are never silently resumed or overwritten.
    """

    def __init__(self, cfg: ModelConfig, output: Path, max_calls: int = 50,
                 temperature: float = 0, *, expected_model: str | None = None,
                 stream: bool = False):
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.cfg = cfg
        self.config = cfg
        self.output = Path(output)
        self.max_calls = max_calls
        self.temperature = temperature
        self.stream = stream
        self.calls: list[dict] = []
        self.returned_model = expected_model
        self._stopped = False
        self._lock = threading.Lock()
        self.output.mkdir(parents=True, exist_ok=True)
        self.call_root = self.output / "api_calls"
        if (self.output / "ledger.json").exists() or (self.call_root.exists() and any(self.call_root.iterdir())):
            raise ValueError("API recording directory already contains a run; choose a fresh output")
        self.call_root.mkdir(exist_ok=True)
        self._save()

    @property
    def public_config(self) -> dict:
        return {**self.cfg.public(), "role": "shared", "temperature": self.temperature}

    def public(self) -> dict:
        return self.public_config

    def _save(self) -> None:
        _write(self.output / "ledger.json", {
            "config": self.public_config, "max_calls": self.max_calls,
            "transport": {"stream": self.stream},
            "limit_unit": "request_attempts", "dollar_cap_enforced": False,
            "request_count": len(self.calls), "returned_model": self.returned_model,
            "stopped": self._stopped, "calls": self.calls,
        })

    def scoped(self, label: str):
        def call(prompt: str, images: list[Path] | None = None) -> ModelResult:
            return self._call(label, prompt, images or [])
        return call

    def __call__(self, prompt: str, images: list[Path] | None = None) -> ModelResult:
        return self._call("shared", prompt, images or [])

    def _call(self, label: str, prompt: str, images: list[Path]) -> ModelResult:
        with self._lock:
            if self._stopped:
                raise ModelError("Frozen API stopped after a previous failure; no automatic retry")
            if len(self.calls) >= self.max_calls:
                raise ModelError("Shared API request-count limit reached; no request sent")
            index = len(self.calls) + 1
            directory = self.call_root / f"{index:03d}"
            directory.mkdir()
            entry = {"index": index, "scope": label, "directory": str(directory.relative_to(self.output)),
                     "started_at": datetime.now(timezone.utc).isoformat(), "status": "started",
                     "usage": {}, "provider_cost": None}
            self.calls.append(entry)
            self._save()  # Count durably before preparing or sending the request.
            start = time.perf_counter()
            try:
                (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
                copied_images, manifest = [], []
                for number, source in enumerate(images):
                    source = Path(source)
                    destination = directory / f"image_{number:03d}{source.suffix.lower()}"
                    shutil.copyfile(source, destination)
                    copied_images.append(destination)
                    manifest.append({"name": destination.name, "source_name": source.name,
                                     "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
                _write(directory / "images.json", manifest)
                result = complete(self.cfg, prompt, copied_images, trace_dir=directory,
                                  temperature=self.temperature, stream=self.stream)
                entry.update(usage=result.usage, returned_model=result.model,
                             finish_reason=result.finish_reason)
                if not result.model:
                    raise ModelError("Provider omitted returned model identity; cannot freeze model")
                if self.returned_model is None:
                    self.returned_model = result.model
                elif result.model != self.returned_model:
                    raise ModelError("Provider returned a different model identity; frozen API stopped")
                entry["status"] = "complete"
                return result
            except Exception as exc:
                self._stopped = True
                entry.update(status="failed", error_type=type(exc).__name__)
                if not (directory / "error.json").exists():
                    _write(directory / "error.json", {"type": type(exc).__name__, "http_code": None})
                raise
            finally:
                metadata = directory / "metadata.json"
                if metadata.exists():
                    details = json.loads(metadata.read_text(encoding="utf-8"))
                    entry["provider_cost"] = details.get("provider_cost")
                    entry["usage"] = details.get("usage", entry["usage"])
                entry["elapsed_ms"] = (time.perf_counter() - start) * 1000
                self._save()
