"""Repair table runner: every judge condition x one frozen generator over a directory of cases, cached and resumable.

Case directory layout (as scripts/prepare_repair_experiment.py writes, one folder per case):
  initial.ts            candidate program the generator must repair
  reference.private.ts  verified reference program (audit and sweep renders only; never shown to the generator)
  case.private.json     {"framing": {center, radius} | null, "tier": easy|medium|hard,
                         "sweep": {"views": [...], "joints": [...], "states": [...]}  (optional, defaults from tier),
                         "annotation": {"id", "labels": [{defect_id, parts, severity, params{joint}}], "grid", "exposures", ...}}
Output: <out>/<generator_id>/<judge_id>/<case_id>/ with the run_repair ledger, plus judge_call.json / verdict.json per round.
An episode whose generator call fails (ModelError) ends with result.json status generator_error and is rerun on resume;
the process then exits non-zero after all cases. TIO_GENERATOR_RETRY_BUDGET_S (default 1800) bounds retries per call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.models.client import FINISHED, ModelResult, TransientModelError, complete
from turnitover.models.config import model_config, read_environment
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.repair.environment import RepairEnvironment
from turnitover.repair.judges import PACK_FORMAT, Case, judge_from_id, load_pack
from turnitover.repair.loop import run_repair
from turnitover.repair.patches import PLACEHOLDER, elide


FIRST_WAIT_S, MAX_WAIT_S = 20, 300


def complete_with_retry(config, prompt: str, images: list[Path]) -> ModelResult:
    """Retry HTTP 429/500/502/503/504 and dropped or timed-out connections (a timed-out call may already be billed).
    Wait doubles from 20 s, at least Retry-After, at most 300 s; give up once the next wait would pass the budget."""
    budget = float(os.environ.get("TIO_GENERATOR_RETRY_BUDGET_S", "1800"))
    start = time.monotonic()
    backoff = FIRST_WAIT_S
    while True:
        try:
            return complete(config, prompt, images)
        except TransientModelError as exc:
            wait = min(MAX_WAIT_S, max(backoff, exc.retry_after or 0))
            elapsed = time.monotonic() - start
            if elapsed + wait > budget:
                raise
            print(json.dumps({"generator_retry": exc.reason, "wait_s": wait, "elapsed_s": round(elapsed, 1)}), flush=True)
            time.sleep(wait)
            backoff = min(MAX_WAIT_S, 2 * backoff)


def cached_api_generator(config, generator_id: str, cache: Path):
    folder = cache / "generator" / generator_id
    folder.mkdir(parents=True, exist_ok=True)
    def generator(prompt: str, images: list[Path]) -> ModelResult:
        digest = hashlib.sha256()
        digest.update(generator_id.encode()); digest.update(b"\0"); digest.update(prompt.encode())
        for path in images:
            digest.update(b"\0"); digest.update(hashlib.sha256(path.read_bytes()).digest())
        key = folder / f"{digest.hexdigest()}.json"
        if key.exists():
            data = json.loads(key.read_text())
            return ModelResult(data["text"], data["model"], data["usage"], data["finish_reason"], 0.0)
        result = complete_with_retry(config, prompt, images)
        if result.finish_reason in FINISHED:   # a truncated response is not cached, so a rerun calls the model again
            key.write_text(json.dumps({"text": result.text, "model": result.model, "usage": result.usage,
                                       "finish_reason": result.finish_reason, "elapsed_ms": result.elapsed_ms,
                                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "images": [p.name for p in images]}))
        return result
    return generator


def scripted_generator(kind: str, case: Case):
    """Not a model. scripted-stop stops at once; scripted-fix rewrites every text segment between geometry placeholders
    that differs from the reference (in the pack both programs hold the same arrays in the same order), then stops."""
    state = {"calls": 0}
    segments = zip(PLACEHOLDER.split(elide(case.initial)[0]), PLACEHOLDER.split(elide(case.reference.source)[0]), strict=True)
    edits = [{"old": old, "new": new} for old, new in segments if old != new]
    def generator(prompt: str, images: list[Path]) -> ModelResult:
        state["calls"] += 1
        if kind == "scripted-fix" and state["calls"] == 1:
            body = {"edits": edits}
        else:
            body = {"stop": True}
        return ModelResult(json.dumps(body), kind, {"input_tokens": 0, "output_tokens": 0}, "completed", 0.0)
    return generator


def episode_done(folder: Path) -> bool:
    result = folder / "result.json"
    return result.exists() and json.loads(result.read_text()).get("status") == "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/generate_replicacad.yaml"))
    parser.add_argument("--judges", nargs="+", required=True,
                        help="none pixel zs8b_screenshot zs8b_sweep ours_2b_ordinary ours_2b_evidence ours_8b_evidence gold")
    parser.add_argument("--generator", default="api", help="api | scripted-stop | scripted-fix")
    parser.add_argument("--generator-model", help="model id for the api generator, overrides TIO_GENERATOR_MODEL")
    parser.add_argument("--generator-id", help="name used in output paths; default derived from the model id")
    parser.add_argument("--model-root", type=Path, help="directory with Qwen3-VL-2B-Instruct and Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter", action="append", default=[], help="judge_id=/path/to/checkpoint")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--default-views", nargs="+", default=["front"])
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--labels", type=Path, help="annotations.private.jsonl of the 900 export (pack input only)")
    parser.add_argument("--episodes", type=Path, help="validation.episodes.jsonl of the 900 export, for per-cell exposure (pack input only)")
    args = parser.parse_args()

    adapters = {k: Path(v) for k, v in (item.split("=", 1) for item in args.adapter)}
    judges = [judge_from_id(j, args.model_root, adapters) for j in args.judges]
    cfg = load_generate_config(args.config)
    pack_manifest = args.cases / "manifest.json"
    if pack_manifest.exists() and json.loads(pack_manifest.read_text()).get("format") == PACK_FORMAT:
        cases = sorted(load_pack(args.cases, args.labels, args.episodes), key=lambda c: c.uid)
    else:
        folders = sorted(p for p in args.cases.iterdir() if p.is_dir() and (p / "initial.ts").exists())
        cases = [Case.load(p, tuple(args.default_views)) for p in folders]
    if args.case_ids:
        cases = [c for c in cases if c.uid in set(args.case_ids)]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no cases found")

    if args.generator == "api":
        if args.generator_model:
            os.environ["TIO_GENERATOR_MODEL"] = args.generator_model
        config = model_config("generator", read_environment(args.env_file))
        config.validate()
        generator_id = args.generator_id or config.model.replace("/", "_")
        generator_meta = config.public()
    elif args.generator in ("scripted-stop", "scripted-fix"):
        config, generator_id, generator_meta = None, args.generator_id or args.generator, {"model": args.generator, "not_a_model": True}
    else:
        raise SystemExit("unknown generator")

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "cache"
    generator_errors = 0
    manifest = {"status": "running", "generator": generator_meta, "generator_id": generator_id, "rounds": args.rounds,
                "config": str(args.config), "cases": [c.uid for c in cases], "judges": [j.describe() for j in judges],
                "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (args.out / f"manifest-{generator_id}.json").write_text(json.dumps(manifest, indent=2))
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
        for judge in judges:
            ledger = args.out / generator_id / judge.judge_id / "episodes.jsonl"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            for case in cases:
                folder = ledger.parent / case.uid
                if episode_done(folder):
                    print(json.dumps({"judge": judge.judge_id, "case": case.uid, "skipped": "complete"}), flush=True)
                    continue
                if folder.exists():
                    folder.rename(folder.with_name(f"{case.uid}.partial-{int(time.time())}"))
                environment = RepairEnvironment(session, case.reference, case.framing, cfg,
                                                {"labels": case.labels, "trajectories": {}})
                generator = (cached_api_generator(config, generator_id, cache) if config
                             else scripted_generator(args.generator, case))
                t0 = time.time()
                result = run_repair(case.initial, generator, judge.bind(session, case, cache), environment.audit,
                                    folder, max_rounds=args.rounds)
                generator_errors += result["status"] == "generator_error"
                row = {"judge": judge.judge_id, "case": case.uid, "tier": case.tier,
                       "labels": [l["defect_id"] for l in case.labels], "seconds": round(time.time() - t0, 1),
                       **{k: result.get(k) for k in ("status", "error", "final_passed", "first_success_round", "model_calls",
                                                     "termination", "non_improving_rate", "exact_state_revisit_rate",
                                                     "total_tokens")}}
                with ledger.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    manifest.update(status="complete" if not generator_errors else "generator_error", generator_errors=generator_errors,
                    finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (args.out / f"manifest-{generator_id}.json").write_text(json.dumps(manifest, indent=2))
    if generator_errors:
        raise SystemExit(f"{generator_errors} episodes ended in generator_error; resubmit to rerun them")


if __name__ == "__main__":
    main()
