#!/usr/bin/env python3
"""Run and score one frozen UniPhys verifier benchmark split."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.models.config import model_config, read_environment
from turnitover.render.session import RenderConfig
from turnitover.render.views import load_views
from turnitover.telemetry import git_state, now_iso
from turnitover.verifier.contracts import VERIFIER_SCHEMA_VERSION
from turnitover.verifier.runner import VerifyConfig, verify
from turnitover.verifier.scoring import score_verifier


SPLITS = ("train", "validation", "test")
TERMINAL_STATUSES = {"complete", "error"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_inside(root: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {label}")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Invalid {label}: {value}")
    return path


def _load_gold(path: Path, split: str) -> dict[str, dict]:
    rows = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id):
            raise ValueError(f"Invalid gold case id on line {number}")
        if case_id in rows:
            raise ValueError(f"Duplicate gold case id: {case_id}")
        if row.get("split") != split or not isinstance(row.get("split_group"), str):
            raise ValueError(f"Gold case is not in the {split} split: {case_id}")
        rows[case_id] = row
    return rows


def _load_benchmark(benchmark: Path, split: str = "test") -> tuple[dict, list[dict], Path]:
    if split not in SPLITS:
        raise ValueError(f"Unsupported benchmark split: {split}")
    root = benchmark.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if (manifest.get("status") != "complete" or not isinstance(cases, list)
            or manifest.get("case_count") != len(cases)
            or manifest.get("preflight", {}).get("validated_cases") != len(cases)):
        raise ValueError("Expected a complete, fully preflighted benchmark")
    if manifest.get("verifier_schema_version") != VERIFIER_SCHEMA_VERSION:
        raise ValueError("Benchmark verifier schema does not match this checkout")
    object_counts, case_counts = manifest.get("object_counts"), manifest.get("case_counts")
    if isinstance(object_counts, dict) and isinstance(case_counts, dict):
        object_count, case_count = object_counts.get(split), case_counts.get(split)
    elif split == "test":
        object_count, case_count = manifest.get("object_count"), manifest.get("case_count")
    else:
        raise ValueError(f"Benchmark manifest has no counts for the {split} split")
    if (type(object_count) is not int or object_count < 1
            or type(case_count) is not int or case_count != 3 * object_count):
        raise ValueError(f"Invalid object/case counts for the {split} split")

    catalog_value = manifest.get("catalog")
    if not isinstance(catalog_value, str) or not catalog_value:
        raise ValueError("Benchmark manifest is missing its catalog")
    catalog = Path(catalog_value)
    if not catalog.is_absolute():
        catalog = (root / catalog).resolve()
    if not catalog.is_file() or _sha256(catalog) != manifest.get("catalog_sha256"):
        raise ValueError("Frozen catalog is missing or its SHA-256 has changed")
    splits_value = manifest.get("splits")
    if splits_value is not None:
        splits_path = Path(splits_value)
        if not splits_path.is_absolute():
            splits_path = (root / splits_path).resolve()
        if not splits_path.is_file() or _sha256(splits_path) != manifest.get("splits_sha256"):
            raise ValueError("Frozen split manifest is missing or its SHA-256 has changed")

    gold_path = root / f"{split}.gold.jsonl"
    gold = _load_gold(gold_path, split)
    if len(gold) != case_count:
        raise ValueError(f"Expected {case_count} unique {split} gold cases, got {len(gold)}")

    loaded = []
    seen = set()
    program_shas = set()
    variants_by_group = defaultdict(set)
    for row in (row for row in cases if row.get("split") == split):
        case_id = row.get("id")
        if (not isinstance(case_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id)
                or case_id in seen):
            raise ValueError(f"Invalid or duplicate benchmark case id: {case_id}")
        seen.add(case_id)
        target = gold.get(case_id)
        if target is None:
            raise ValueError(f"Benchmark case is missing from gold: {case_id}")
        if (row.get("split_group") != target.get("split_group") or target.get("split") != split):
            raise ValueError(f"Case split identity mismatch: {case_id}")
        program_path = _resolve_inside(root, row.get("program"), f"program for {case_id}")
        reference_program_path = _resolve_inside(
            root, row.get("reference_program"), f"reference program for {case_id}")
        references = row.get("reference_images")
        if not isinstance(references, list) or not references:
            raise ValueError(f"Missing reference images for {case_id}")
        reference_paths = tuple(
            _resolve_inside(root, value, f"reference image for {case_id}") for value in references)
        program = ObjectProgram(program_path.read_text(encoding="utf-8"))
        reference_program = ObjectProgram(reference_program_path.read_text(encoding="utf-8"))
        if program.sha != target.get("program_sha") or reference_program.sha != target.get("reference_sha"):
            raise ValueError(f"Frozen program SHA mismatch: {case_id}")
        if program.sha in program_shas:
            raise ValueError(f"Duplicate candidate program SHA: {case_id}")
        program_shas.add(program.sha)
        variant = case_id.rsplit("-", 1)[-1]
        variants_by_group[row["split_group"]].add(variant)
        loaded.append({
            "id": case_id,
            "task": row.get("task", "Verify this articulated object."),
            "split_group": row["split_group"],
            "program": program,
            "program_path": program_path,
            "reference_program": reference_program,
            "reference_program_path": reference_program_path,
            "references": reference_paths,
        })
    if seen != set(gold):
        raise ValueError("Benchmark and gold case IDs differ")
    if (len(variants_by_group) != object_count
            or any(variants != {"clean", "axis", "offset"}
                   for variants in variants_by_group.values())):
        raise ValueError(
            f"Expected clean/axis/offset exactly once for each of {object_count} split groups")
    return manifest, loaded, gold_path


def _totals(tasks: list[dict]) -> dict:
    return {
        "planned": len(tasks),
        "completed": sum(task["status"] in TERMINAL_STATUSES for task in tasks),
        "errors": sum(task["status"] in {"error", "runner_error"} for task in tasks),
    }


def run(
    benchmark: Path,
    output: Path,
    env_file: Path,
    *,
    split: str = "test",
    policy: str = "active",
    budget: int = 8,
    seed: int = 0,
    size: int = 384,
    views_path: Path = REPO_ROOT / "configs/views.yaml",
    action_types: tuple[str, ...] = ("request_view", "actuate_joint", "query_runtime"),
    max_triangles: int = 1_000_000,
    max_draw_calls: int = 1_024,
    local_model: Path | None = None,
    verify_case: Callable = verify,
    score: Callable = score_verifier,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Batch output directory is not empty: {output}")
    benchmark_manifest, cases, gold_path = _load_benchmark(benchmark, split)
    config = VerifyConfig(budget=budget, mode=policy, seed=seed, action_types=action_types,
                          max_triangles=max_triangles, max_draw_calls=max_draw_calls)
    config.validate()
    model = None
    if policy != "runtime" and local_model is None:
        model = model_config("judge", read_environment(env_file))
        model.validate()
    render = RenderConfig(REPO_ROOT / "web/dist", REPO_ROOT / "web/node_modules/.bin/esbuild", size, size)
    views = load_views(views_path)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "kind": "uniphys_verifier_batch",
        "status": "running",
        "created_at": now_iso(),
        "git": git_state(REPO_ROOT),
        "benchmark": str(benchmark.resolve()),
        "benchmark_manifest_sha256": _sha256(benchmark.resolve() / "manifest.json"),
        "catalog_sha256": benchmark_manifest["catalog_sha256"],
        "splits_sha256": benchmark_manifest.get("splits_sha256"),
        "model": model.public() if model else {
            "kind": "local_qwen" if local_model else "offline",
            "name": local_model.name if local_model else None,
        },
        "config": {
            "split": split, "policy": policy, "budget": budget, "seed": seed, "size": size,
            "views": str(views_path.resolve()), "action_types": list(action_types),
            "max_triangles": max_triangles, "max_draw_calls": max_draw_calls,
        },
        "tasks": [{
            "id": case["id"], "status": "pending", "split": split,
            "split_group": case["split_group"], "program_sha": case["program"].sha,
            "reference_sha": case["reference_program"].sha,
            "reference_image_sha256": [_sha256(path) for path in case["references"]],
        } for case in cases],
    }
    manifest_path = output / "batch-manifest.json"

    def save() -> None:
        manifest["totals"] = _totals(manifest["tasks"])
        write_json_atomic(manifest_path, manifest)

    save()
    try:
        for index, case in enumerate(cases):
            task = manifest["tasks"][index]
            task["status"] = "running"
            save()
            case_output = output / case["id"]
            try:
                result = verify_case(
                    case["program"], case_output, render, views, config,
                    task=case["task"], references=case["references"], model=model,
                    local_model=local_model, reference_program=case["reference_program"],
                )
                if (result.get("schema_version") != VERIFIER_SCHEMA_VERSION
                        or result.get("program_sha") != case["program"].sha
                        or result.get("status") not in TERMINAL_STATUSES
                        or not (case_output / "result.json").is_file()):
                    raise RuntimeError(f"Invalid verifier result for {case['id']}")
                task.update(status=result["status"], output=str(case_output),
                            result=str(case_output / "result.json"),
                            verdict=result.get("verdict", {}).get("status"),
                            termination=result.get("termination"), spent=result.get("spent"),
                            error_type=result.get("error_type"))
                save()
            except Exception as exc:
                task.update(status="runner_error", output=str(case_output),
                            error_type=type(exc).__name__, error=str(exc))
                save()
                raise
        metrics_path = output / "metrics.json"
        metrics = score(gold_path, output, metrics_path)
        manifest.update(status="complete", metrics=str(metrics_path),
                        metric_status=metrics.get("status"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("data/benchmarks/uniphys-verifier-test"))
    parser.add_argument("--out", type=Path, default=Path("output/uniphys-verifier-test"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--policy", choices=("active", "fixed", "random", "runtime"), default="active")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--views", type=Path, default=Path("configs/views.yaml"))
    parser.add_argument("--actions", nargs="+", default=["request_view", "actuate_joint", "query_runtime"],
                        choices=("request_view", "actuate_joint", "query_runtime"))
    parser.add_argument("--max-triangles", type=int, default=1_000_000)
    parser.add_argument("--max-draw-calls", type=int, default=1_024)
    parser.add_argument("--local-model", type=Path)
    args = parser.parse_args()
    report = run(
        args.benchmark.resolve(), args.out.resolve(), args.env_file.resolve(), split=args.split,
        policy=args.policy,
        budget=args.budget, seed=args.seed, size=args.size, views_path=args.views.resolve(),
        action_types=tuple(args.actions), max_triangles=args.max_triangles,
        max_draw_calls=args.max_draw_calls,
        local_model=args.local_model.resolve() if args.local_model else None,
    )
    print(json.dumps({"status": report["status"], **report["totals"],
                      "manifest": str(args.out.resolve() / "batch-manifest.json")}))


if __name__ == "__main__":
    main()
