"""Run one auditable interactive-teacher episode on a real PartNet Train candidate."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.models.config import model_config, read_environment
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.teacher.audit import audit_episode
from turnitover.teacher.episode import run_teacher_episode
from turnitover.teacher.policy import TeacherPolicy
from turnitover.teacher.providers import KimiPricing, KimiTeacher, LocalQwenTeacher, MockTeacher
from turnitover.verifier.contracts import Context
from turnitover.config import load_generate_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/benchmarks/partnet-mobility-formal-v1/prepared"
DEFAULT_CONFIG = ROOT / "configs/generate_partnet_mobility_supervision.yaml"
DEFAULT_MODEL = ROOT.parent / "TurnItOver/models/Qwen3-VL-2B-Instruct"


def _rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}


def _sample(benchmark: Path, sample_id: str | None) -> tuple[dict, dict]:
    inputs = _rows(benchmark / "train.inputs.jsonl")
    gold = _rows(benchmark / "train.gold.jsonl")
    uid = sample_id or next(iter(inputs))
    if uid not in inputs or uid not in gold:
        raise ValueError("Sample must identify one prepared PartNet Train row")
    if gold[uid].get("provenance", {}).get("formal_training") is not True:
        raise ValueError("Interactive teacher requires a formal Train asset")
    return inputs[uid], gold[uid]


def _mock(context: Context) -> MockTeacher:
    if context.joints:
        action = {"action": {"type": "actuate_joint", "joint_id": context.joints[0], "detent": "limit"}}
    else:
        action = {"action": {"type": "request_view", "view_id": context.views[0]}}
    verdict = {"verdict": {"status": "uncertain", "confidence": 0.0, "findings": [],
               "summary": "The bounded observation does not establish a complete diagnosis.",
               "limitations": ["Only the acquired public observation was considered."]}}
    return MockTeacher([action, verdict])


def _skip(output: Path, reason: str, model: str | None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Kimi preflight output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "status": "skipped", "phase": "kimi",
              "real_api_requests": 0, "actual_cost_rmb": 0.0, "model": model,
              "reason": reason}
    write_json_atomic(output / "episode.json", result)
    write_json_atomic(output / "audit.json", {"pipeline_ok": False, "teacher_actions_legal": None,
                                               "diagnosis_evidence_supported": None,
                                               "skip_reason": reason})
    return result


def run(args) -> dict:
    if args.playwright_site_packages:
        sys.path.append(str(args.playwright_site_packages.resolve()))
    public, private = _sample(args.benchmark, args.sample_id)
    context = Context(**public["context"])
    cfg = load_generate_config(args.config)
    if args.mode == "mock":
        teacher = _mock(context)
    elif args.mode == "local-qwen":
        teacher = LocalQwenTeacher(args.local_model, args.max_tokens)
    else:
        env = read_environment(args.env_file)
        model = model_config("teacher", env)
        missing = []
        if not args.third_party_permission_confirmed:
            missing.append("asset/image permission for third-party transmission is not confirmed")
        if args.estimated_input_tokens is None:
            missing.append("input-token estimate is missing")
        if None in (args.input_price_rmb, args.output_price_rmb) or not args.pricing_verified_at or not args.pricing_source:
            missing.append("current pricing and its source are not fully verified")
        try:
            model.validate()
        except ValueError as exc:
            missing.append(str(exc))
        if missing:
            return _skip(args.out, "; ".join(missing), model.model or None)
        pricing = KimiPricing(args.input_price_rmb, args.output_price_rmb,
                              args.pricing_verified_at, args.pricing_source)
        teacher = KimiTeacher(model, pricing, permission_confirmed=True,
                              estimated_input_tokens=args.estimated_input_tokens,
                              max_cost_rmb=args.max_cost_rmb)
        print(json.dumps({"phase": "kimi_preflight", "model": model.model,
                          "estimated_input_tokens": args.estimated_input_tokens,
                          "max_output_tokens": model.max_tokens,
                          "estimated_max_cost_rmb": teacher.estimated_cost_rmb,
                          "real_request_cap": 1}), flush=True)

    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
        references = []
        for index, relative in enumerate(public["reference_images"]):
            source = args.benchmark / relative
            target = args.out / f"reference-{index:02d}{source.suffix}"
            shutil.copyfile(source, target)
            references.append(target)
    else:
        references = sorted(args.out.glob("reference-*"))
    policy = TeacherPolicy(teacher, args.out, references)
    program = ObjectProgram(private["candidate_program"])
    max_calls = 1 if args.mode == "kimi" else None
    if args.mode == "kimi" and args.resume:
        saved = json.loads((args.out / "episode.json").read_text())
        if saved.get("model_calls"):
            raise ValueError("A paid Kimi episode with a recorded call cannot be resumed or retried")
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
        session.load(program, framing=None)
        episode = run_teacher_episode(session, policy, context, args.budget, args.out,
                                      episode_id=public["id"], resume=args.resume,
                                      max_model_calls=max_calls,
                                      decision_attempts=1 if args.mode == "kimi" else 3)
    if args.mode == "kimi":
        usage = episode["token_usage"]
        episode["real_api_requests"] = teacher.calls
        episode["actual_cost_rmb"] = teacher.pricing.estimate(usage["input_tokens"], usage["output_tokens"])
        write_json_atomic(args.out / "episode.json", episode)
    audit_episode(args.out, private)
    return episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "local-qwen", "kimi"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-id")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--local-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--playwright-site-packages", type=Path,
                        help="Append an existing Playwright install when the Qwen environment lacks it")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--third-party-permission-confirmed", action="store_true")
    parser.add_argument("--estimated-input-tokens", type=int)
    parser.add_argument("--input-price-rmb", type=float)
    parser.add_argument("--output-price-rmb", type=float)
    parser.add_argument("--pricing-verified-at")
    parser.add_argument("--pricing-source")
    parser.add_argument("--max-cost-rmb", type=float, default=1.0)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments), indent=2))
