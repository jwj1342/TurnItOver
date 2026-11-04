"""One model call that produces a Program-ABI compatible TypeScript candidate."""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from turnitover.core.program import ObjectProgram
from turnitover.core.jsonio import write_json_atomic
from turnitover.models.client import ModelError, ModelResult, image_part, is_normal_finish_reason
from turnitover.models.commands import extract_program


ModelCall = Callable[[str, list[Path]], ModelResult]
@dataclass(frozen=True)
class GenerationResult:
    program: ObjectProgram
    model: str
    usage: dict
    finish_reason: str
    elapsed_ms: float


def generate_program(prompt: str, images: tuple[Path, ...], output: Path, call: ModelCall) -> GenerationResult:
    """Generate one candidate and retain the exact request/response without credentials."""
    if not prompt.strip():
        raise ValueError("Generation prompt must be nonempty")
    for path in images:
        image_part(path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Generation output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    copied: list[Path] = []
    image_records = []
    for index, path in enumerate(images):
        target = output / f"input-{index:02d}{path.suffix.lower()}"
        content = path.read_bytes()
        target.write_bytes(content)
        copied.append(target)
        image_records.append({"path": target.name, "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"version": 1, "status": "running", "images": image_records,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    write_json_atomic(output / "result.json", manifest)
    try:
        response = call(prompt, copied)
        (output / "response.txt").write_text(response.text, encoding="utf-8")
        public = {key: value for key, value in dataclasses.asdict(response).items() if key != "text"}
        write_json_atomic(output / "model.json", public)
        if not is_normal_finish_reason(response.finish_reason):
            raise ModelError("Incomplete generator response")
        program = ObjectProgram(extract_program(response.text))
        (output / "program.ts").write_text(program.source, encoding="utf-8")
        manifest.update(status="complete", program_sha=program.sha, model=public)
        write_json_atomic(output / "result.json", manifest)
        return GenerationResult(program, response.model, response.usage, response.finish_reason, response.elapsed_ms)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json_atomic(output / "result.json", manifest)
        raise
