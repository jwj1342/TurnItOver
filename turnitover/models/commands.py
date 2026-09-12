"""CLI workflows; credentials never enter output metadata."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

from turnitover.models.client import complete
from turnitover.models.config import ROLES, model_config, read_environment
from turnitover.telemetry import git_state, now_iso


def check_models(args) -> int:
    env = read_environment(args.env_file)
    rows = []
    for role in ROLES:
        try:
            cfg = model_config(role, env)
            row = cfg.public()
            try:
                cfg.validate()
                row["ready"] = True
            except ValueError as exc:
                row.update(ready=False, error=str(exc))
        except ValueError as exc:
            row = {"role": role, "ready": False, "error": str(exc)}
        rows.append(row)
    print(json.dumps(rows, indent=2))
    return 0 if all(row["ready"] for row in rows) else 1


def run_call(args, repo_root: Path, *, reconstruct: bool = False) -> int:
    cfg = model_config(args.role, read_environment(args.env_file))
    if not args.dry_run:
        cfg.validate()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    # Validate media before creating an output directory, including on dry runs.
    from turnitover.models.client import image_part

    images = args.image or []
    for image in images:
        image_part(image)
    if reconstruct and not images:
        raise ValueError("Reconstruction requires at least one --image")
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "prepared", "created_at": now_iso(), "model_config": cfg.public(),
                "task": "single_pass_reconstruction" if reconstruct else "model_call",
                "git": git_state(repo_root), "images": [],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    for i, image in enumerate(images):
        # Keep actual inputs for reproducibility; preserve neither credentials nor source paths.
        name = f"input-{i:02d}{image.suffix.lower()}"
        content = image.read_bytes()
        (output / name).write_bytes(content)
        manifest["images"].append({"path": name, "sha256": hashlib.sha256(content).hexdigest()})
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save()
    if args.dry_run:
        print(f"Prepared inputs without an API call: {output}")
        return 0
    try:
        result = complete(cfg, prompt, images)
        (output / "response.txt").write_text(result.text, encoding="utf-8")
        manifest["response"] = {k: v for k, v in dataclasses.asdict(result).items() if k != "text"}
        manifest["status"] = "responded"
        if reconstruct:
            if result.finish_reason not in {"completed", "end_turn", "STOP", "stop"}:
                raise ValueError("Model response did not finish normally; inspect response.txt before using it")
            source = extract_program(result.text)
            (output / "program.ts").write_text(source, encoding="utf-8")
            from turnitover.render.transpile import transpile_ts

            transpile_ts(source, repo_root / "web/node_modules/.bin/esbuild")
            manifest["program_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            manifest["syntax_checked"] = True
            manifest["runtime_validated"] = False
        manifest.update(status="complete", finished_at=now_iso())
        save()
    except Exception as exc:
        # Compile errors may include generated text, so store only the exception type here.
        manifest.update(status="failed", error_type=type(exc).__name__)
        save()
        raise
    print(output)
    return 0


def extract_program(text: str) -> str:
    blocks = re.findall(r"```(?:typescript|ts|javascript|js)?\s*\n(.*?)```", text, re.DOTALL)
    if len(blocks) > 1:
        raise ValueError("Expected one program code block; inspect response.txt")
    source = blocks[0].strip() if blocks else text.strip()
    if "export default" not in source:
        raise ValueError("Response has no default export; inspect response.txt")
    return source + "\n"
