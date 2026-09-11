"""TypeScript -> ES module JavaScript using the esbuild binary from web/node_modules."""
from __future__ import annotations

import subprocess
from pathlib import Path


class CompileError(Exception):
    pass


def transpile_ts(source: str, esbuild_bin: Path) -> str:
    cmd = [str(esbuild_bin), "--loader=ts", "--format=esm", "--target=es2022", "--log-level=error"]
    proc = subprocess.run(cmd, input=source, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CompileError(proc.stderr.strip() or f"esbuild exited {proc.returncode}")
    return proc.stdout
