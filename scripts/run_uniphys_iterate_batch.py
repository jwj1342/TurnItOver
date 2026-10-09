#!/usr/bin/env python3
"""Run one image-to-program ``turnitover iterate`` task per UniPhys object."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.core.jsonio import write_json_atomic
from turnitover.models.config import model_config, read_environment
from turnitover.telemetry import git_state, now_iso

TERMINAL_STATUSES = {
    "accepted",
    "max_visual_revisions",
    "render_failed",
    "verification_uncertain",
    "verification_error",
    "generation_failed",
}
TASK_TEXT = "Reconstruct and verify the articulated object shown in the reference image."


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_snapshot(env_file: Path, expected_model: str) -> dict:
    environment = read_environment(env_file)
    result = {}
    for role in ("generator", "judge"):
        config = model_config(role, environment)
        config.validate()
        if config.model != expected_model:
            raise ValueError(f"Expected {role} model {expected_model!r}, got {config.model!r}")
        result[role] = config.public()
    return result


def _load_inputs(references: Path, view: str) -> list[dict]:
    source = json.loads((references / "manifest.json").read_text(encoding="utf-8"))
    if source.get("status") != "complete" or len(source.get("objects", [])) != 20:
        raise ValueError("Expected a complete 20-object UniPhys reference manifest")
    tasks = []
    for row in source["objects"]:
        object_id = row["object_id"]
        image = references / object_id / "screenshots" / f"{view}.png"
        if not image.is_file():
            raise FileNotFoundError(f"Missing reference image: {image}")
        tasks.append({"object_id": object_id, "image": image.resolve(), "sha256": _sha256(image)})
    if len({row["object_id"] for row in tasks}) != len(tasks):
        raise ValueError("Duplicate object IDs in reference manifest")
    return tasks


def _totals(tasks: list[dict]) -> dict:
    completed = [task for task in tasks if task["status"] in TERMINAL_STATUSES]
    calls = {key: 0 for key in ("generation", "runtime_repair", "visual_revision", "judge")}
    for task in completed:
        for key in calls:
            calls[key] += task.get("model_calls", {}).get(key, 0)
    return {
        "planned": len(tasks),
        "completed": len(completed),
        "accepted": sum(task.get("accepted") is True for task in completed),
        "model_calls": calls,
        "total_tokens": sum(task.get("total_tokens", 0) for task in completed),
    }


def _task_record(task: dict, result: dict, returncode: int, output: Path) -> dict:
    copied_references = result.get("references", [])
    copied_sha = copied_references[0].get("sha256") if len(copied_references) == 1 else None
    return {
        **task,
        "image": str(task["image"]),
        "output": str(output),
        "returncode": returncode,
        "status": result.get("status"),
        "accepted": result.get("accepted"),
        "termination": result.get("termination"),
        "error_type": result.get("error_type"),
        "rounds": len(result.get("rounds", [])),
        "model_calls": result.get("model_calls", {}),
        "total_tokens": result.get("total_tokens", 0),
        "copied_reference_sha256": copied_sha,
        "input_sha_matched": copied_sha == task["sha256"],
        "result": str(output / "result.json"),
    }


def run(
    references: Path,
    output: Path,
    env_file: Path,
    *,
    expected_model: str = "qwen3.7-plus",
    view: str = "06",
    budget: int = 8,
    seed: int = 0,
    size: int = 384,
    max_runtime_repairs: int = 3,
    max_visual_revisions: int = 2,
    execute: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Batch output directory is not empty: {output}")
    models = _model_snapshot(env_file, expected_model)
    inputs = _load_inputs(references, view)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "kind": "uniphys_iterate_batch",
        "status": "running",
        "created_at": now_iso(),
        "git": git_state(REPO_ROOT),
        "models": models,
        "config": {
            "reference_view": view,
            "task": TASK_TEXT,
            "policy": "active",
            "budget": budget,
            "seed": seed,
            "size": size,
            "max_runtime_repairs": max_runtime_repairs,
            "max_visual_revisions": max_visual_revisions,
        },
        "tasks": [{**row, "image": str(row["image"]), "status": "pending"} for row in inputs],
    }
    manifest_path = output / "batch-manifest.json"

    def save() -> None:
        manifest["totals"] = _totals(manifest["tasks"])
        write_json_atomic(manifest_path, manifest)

    save()
    try:
        for index, task in enumerate(inputs):
            task_output = output / task["object_id"]
            manifest["tasks"][index]["status"] = "running"
            save()
            command = [
                sys.executable,
                "-m", "turnitover", "iterate",
                "--reference-image", str(task["image"]),
                "--out", str(task_output),
                "--task", TASK_TEXT,
                "--policy", "active",
                "--budget", str(budget),
                "--seed", str(seed),
                "--size", str(size),
                "--max-runtime-repairs", str(max_runtime_repairs),
                "--max-visual-revisions", str(max_visual_revisions),
                "--env-file", str(env_file),
            ]
            completed = execute(command, cwd=REPO_ROOT, capture_output=True, text=True)
            task_output.mkdir(parents=True, exist_ok=True)
            (task_output / "runner.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
            (task_output / "runner.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
            result_path = task_output / "result.json"
            if not result_path.is_file():
                manifest["tasks"][index].update(
                    status="runner_error", returncode=completed.returncode,
                    output=str(task_output), error="iterate did not produce result.json")
                raise RuntimeError(f"iterate produced no result.json for {task['object_id']}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            record = _task_record(task, result, completed.returncode, task_output)
            manifest["tasks"][index] = record
            save()
            if record["status"] not in TERMINAL_STATUSES:
                raise RuntimeError(f"Non-terminal result for {task['object_id']}: {record['status']}")
            if not record["input_sha_matched"]:
                raise RuntimeError(f"Copied reference hash mismatch for {task['object_id']}")
            expected_returncode = 0 if record["accepted"] is True else 1
            if completed.returncode != expected_returncode:
                raise RuntimeError(
                    f"Unexpected iterate exit code for {task['object_id']}: "
                    f"{completed.returncode}, expected {expected_returncode}")
        manifest["status"] = "complete"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest["finished_at"] = now_iso()
        save()
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path("data/references/uniphys"))
    parser.add_argument("--out", type=Path, default=Path("output/uniphys-qwen37"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--expected-model", default="qwen3.7-plus")
    parser.add_argument("--view", default="06", help="two-digit static screenshot index")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--max-runtime-repairs", type=int, default=3)
    parser.add_argument("--max-visual-revisions", type=int, default=2)
    args = parser.parse_args()
    report = run(args.references.resolve(), args.out.resolve(), args.env_file.resolve(),
                 expected_model=args.expected_model, view=args.view, budget=args.budget,
                 seed=args.seed, size=args.size, max_runtime_repairs=args.max_runtime_repairs,
                 max_visual_revisions=args.max_visual_revisions)
    print({"status": report["status"], **report["totals"], "manifest": args.out / "batch-manifest.json"})
