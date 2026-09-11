"""Structured logging, phase timing and run manifests. stdlib only."""
from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_STD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        out.update({k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False, default=str)


def configure_logging(json_lines: bool = False, level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_lines else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


@contextmanager
def timed(timing: dict[str, float], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timing[key] = timing.get(key, 0.0) + (time.perf_counter() - t0) * 1000.0


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def git_state(repo: Path) -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout.strip())
        return {"sha": sha or None, "dirty": dirty}
    except OSError:
        return {"sha": None, "dirty": None}


def environment_info() -> dict:
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def summarize_timings(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    import numpy as np

    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=float)
        out[k] = {"mean_ms": float(vals.mean()), "p95_ms": float(np.percentile(vals, 95)), "n": int(len(vals))}
    return out
