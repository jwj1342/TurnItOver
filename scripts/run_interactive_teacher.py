"""Run one auditable interactive-teacher episode on a real PartNet Train candidate."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from turnitover.core.jsonio import write_json_atomic
from turnitover.core.actions import RuntimeProperty, ViewId
from turnitover.core.program import ObjectProgram
from turnitover.assets.toy import toy_cabinet
from turnitover.models.config import model_config, read_environment
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.teacher.audit import audit_episode
from turnitover.teacher.episode import run_teacher_episode
from turnitover.teacher.policy import TeacherPolicy
from turnitover.teacher.providers import KimiPricing, KimiTeacher, LocalQwenTeacher, MockTeacher
from turnitover.teacher.evidence import bind_witnesses
from turnitover.teacher.witnesses import REFERENCE_PROTOCOL, file_sha256, sample_framing
from turnitover.policy.loop import execute_observation
from turnitover.core.serde import to_dict
from turnitover.verifier.contracts import parse_decision
from turnitover.verifier.contracts import Context
from turnitover.config import load_generate_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/benchmarks/partnet-mobility-formal-v1/prepared"
DEFAULT_CONFIG = ROOT / "configs/generate_partnet_mobility_supervision.yaml"
DEFAULT_ANNOTATIONS = (ROOT / "data/benchmarks/partnet-mobility-formal-v1/"
                       "oracle-first-failure-v4/annotations.private.jsonl")
DEFAULT_MODEL = ROOT.parent / "TurnItOver/models/Qwen3-VL-2B-Instruct"


def _rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}


def _sample(benchmark: Path, sample_id: str | None) -> tuple[dict, dict, object]:
    inputs = _rows(benchmark / "train.inputs.jsonl")
    gold = _rows(benchmark / "train.gold.jsonl")
    uid = sample_id or next(iter(inputs))
    if uid not in inputs or uid not in gold:
        raise ValueError("Sample must identify one prepared PartNet Train row")
    if gold[uid].get("provenance", {}).get("formal_training") is not True:
        raise ValueError("Interactive teacher requires a formal Train asset")
    private = gold[uid]
    framing = sample_framing(benchmark.parent / "raw", private["source_id"], private["sample_id"])
    return inputs[uid], private, framing


def _verify_fixed_protocol(session, program, framing, context, manifest, actions, folder: Path) -> None:
    """Dry-run the fixed history before any paid call; every step must bind to a frozen witness."""
    folder.mkdir(parents=True)
    session.load(program, framing=framing)
    records = []
    for step, raw in enumerate(actions):
        def sink(index, png):
            relative = f"{index:03d}.png"
            (folder / relative).write_bytes(png)
            return relative
        observation = execute_observation(session, parse_decision({"action": raw}, context, set()), step, sink)
        records.append({**to_dict(observation), "decision": raw, "decision_source": "fixed_protocol"})
    rows = bind_witnesses({"context": vars(context), "trajectory": records,
                           "reference_protocol": REFERENCE_PROTOCOL}, manifest, folder)
    write_json_atomic(folder / "binding.json", rows)
    if not all(row.get("bound") and row.get("reference_comparable") for row in rows):
        raise ValueError("Fixed evidence protocol is not fully bound to frozen, reference-comparable witnesses")
    session.load(program, framing=framing)


def _project_fixture(views) -> tuple[dict, dict]:
    """Return a project-authored compatibility fixture whose pixels may be sent externally."""
    spec = toy_cabinet()
    program = ObjectProgram.from_spec(spec)
    context = Context(
        task="Verify the articulated candidate against the reference and configured constraints.",
        parts=spec.part_ids,
        joints=tuple(joint.id for joint in spec.joints),
        views=tuple(str(view.id) for view in views),
        runtime_properties=tuple(item.value for item in RuntimeProperty),
        action_types=("request_view", "actuate_joint", "query_runtime"),
        max_triangles=1_000_000,
        max_draw_calls=1_024,
    )
    identity = {
        "source_id": "turnitover-project-fixture-v1",
        "asset_id": spec.asset_id,
        "split": "train",
        "split_group": "turnitover-project-fixture:toy_cabinet",
        "candidate_sha": program.sha,
        "formal_training": False,
        "purpose": "single_api_compatibility_test_only",
        "outbound_image_permission": True,
        "permission_basis": "project-authored synthetic asset",
    }
    public = {"id": "kimi-compat-toy-cabinet-v1", "context": vars(context),
              "reference_images": [], "data_identity": identity}
    private = {"candidate_program": program.source, "target": {"status": "pass", "findings": []},
               "data_identity": identity}
    return public, private


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
              "real_api_requests": 0, "usage_cost_rmb_estimate": None,
              "billing_cost_rmb": None, "model": model,
              "reason": reason}
    write_json_atomic(output / "episode.json", result)
    write_json_atomic(output / "audit.json", {"pipeline_ok": False, "teacher_actions_legal": None,
                                               "diagnosis_evidence_supported": None,
                                               "skip_reason": reason})
    return result


def run(args) -> dict:
    if args.playwright_site_packages:
        sys.path.append(str(args.playwright_site_packages.resolve()))
    cfg = load_generate_config(args.config)
    views = load_views(cfg.views_path)
    if args.project_fixture:
        public, private = _project_fixture(views)
        framing = None
        annotation = None
    else:
        public, private, framing = _sample(args.benchmark, args.sample_id)
        annotation = _rows(args.oracle_annotations).get(public["id"])
        if annotation is None:
            raise ValueError("Train sample lacks an independent Oracle evidence annotation")
    context = Context(**public["context"])
    manifest, manifest_path = None, None
    if not args.project_fixture:
        if args.frozen_witnesses is None:
            raise ValueError("Real Train episodes require a frozen witness manifest for independent audit")
        manifest_path = args.frozen_witnesses / public["id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
    if args.reference_protocol == REFERENCE_PROTOCOL and manifest is None:
        raise ValueError("The multi-state reference protocol needs a frozen witness manifest")
    if args.experiment_mode == "fixed-diagnosis" and args.reference_protocol != REFERENCE_PROTOCOL:
        raise ValueError("Fixed diagnosis compares motion states and needs the multi-state reference")
    if args.mode == "mock":
        teacher = _mock(context)
    elif args.mode == "local-qwen":
        teacher = LocalQwenTeacher(args.local_model, args.max_tokens)
    else:
        env = read_environment(args.env_file)
        model = model_config("teacher", env)
        missing = []
        if not args.third_party_transmission_reviewed or not args.transmission_review_basis:
            missing.append("asset/image third-party transmission terms and review basis are required")
        if args.estimated_input_tokens is None:
            missing.append("input-token estimate is missing")
        pricing_values = (args.input_price_rmb, args.output_price_rmb,
                          args.pricing_verified_at, args.pricing_source)
        if any(value is not None for value in pricing_values) and not all(
                value is not None for value in pricing_values):
            missing.append("pricing fields must be supplied together or omitted when gateway billing is unknown")
        try:
            model.validate()
        except ValueError as exc:
            missing.append(str(exc))
        if missing:
            return _skip(args.out, "; ".join(missing), model.model or None)
        pricing = (KimiPricing(args.input_price_rmb, args.output_price_rmb,
                               args.pricing_verified_at, args.pricing_source)
                   if all(value is not None for value in pricing_values) else None)
        teacher = KimiTeacher(model, pricing, transmission_reviewed=True,
                              estimated_input_tokens=args.estimated_input_tokens,
                              max_cost_rmb=args.max_cost_rmb,
                              max_real_requests=args.max_real_requests)
        print(json.dumps({"phase": "kimi_preflight", "model": model.model,
                          "estimated_input_tokens": args.estimated_input_tokens,
                          "max_output_tokens": model.max_tokens,
                          "estimated_max_cost_rmb": teacher.estimated_cost_rmb,
                          "real_request_cap": args.max_real_requests}), flush=True)

    program = ObjectProgram(private["candidate_program"])
    max_calls = args.max_real_requests if args.mode == "kimi" else None
    if args.mode == "kimi" and args.resume:
        saved = json.loads((args.out / "episode.json").read_text())
        if saved.get("model_calls"):
            raise ValueError("A paid Kimi episode with a recorded call cannot be resumed or retried")
    data_identity = public.get("data_identity") or {
        "source_id": private.get("source_id"), "asset_id": private.get("asset_id"),
        "split": "train", "split_group": private.get("split_group"),
        "candidate_sha": private.get("program_sha"), "reference_sha": private.get("reference_sha"),
        "formal_training": private.get("provenance", {}).get("formal_training"),
    }
    if annotation is not None:
        data_identity = {**annotation["data_identity"], **data_identity}
    data_identity = {**data_identity, "experiment_mode": args.experiment_mode,
                     "reference_protocol": args.reference_protocol}
    if args.mode == "kimi":
        data_identity = {**data_identity, "third_party_transmission_reviewed": True,
                         "transmission_review_basis": args.transmission_review_basis}
    with ObservationSession(cfg.render, views) as session:
        session.load(program, framing=framing)
        joint_limits = {joint: tuple(row["limits"]) for joint, row in session.info.joints.items()}
        if not args.resume:
            args.out.mkdir(parents=True, exist_ok=False)
            references = []
            reference_states = []
            if args.project_fixture:
                target = args.out / "reference-00.png"
                target.write_bytes(session.request_view(ViewId("front")).png)
                references.append(target)
                reference_states.append({"view": "front", "joint_id": None, "detent": "rest"})
                session.load(program, framing=framing)
            elif args.reference_protocol == REFERENCE_PROTOCOL:
                for row in manifest["reference_protocol"]["states"]:
                    target = args.out / row["file"]
                    shutil.copyfile(manifest_path.parent / row["file"], target)
                    if file_sha256(target) != row["sha256"]:
                        raise ValueError("Frozen reference image changed after freezing")
                    references.append(target)
                    reference_states.append(row["label"])
            else:
                for index, relative in enumerate(public["reference_images"]):
                    source = args.benchmark / relative
                    target = args.out / f"reference-{index:02d}{source.suffix}"
                    shutil.copyfile(source, target)
                    references.append(target)
                    reference_states.append({"view": "front", "joint_id": None, "detent": "rest"})
        else:
            references = sorted(args.out.glob("reference-*"))
            reference_states = ([row["label"] for row in manifest["reference_protocol"]["states"]]
                                if args.reference_protocol == REFERENCE_PROTOCOL else
                                [{"view": "front", "joint_id": None, "detent": "rest"}])
        initial_actions = ()
        if args.experiment_mode == "fixed-diagnosis":
            initial_actions = tuple(manifest["fixed_protocol_actions"])
            if args.budget != len(initial_actions):
                raise ValueError("Fixed diagnosis budget must exactly match its evidence protocol")
            if not args.resume:
                _verify_fixed_protocol(session, program, framing, context, manifest, initial_actions,
                                       args.out / "fixed-protocol-preflight")
        policy = TeacherPolicy(teacher, args.out, references, reference_states=reference_states)
        episode = run_teacher_episode(session, policy, context, args.budget, args.out,
                                      episode_id=public["id"], resume=args.resume,
                                      max_model_calls=max_calls,
                                      decision_attempts=args.decision_attempts,
                                      data_identity=data_identity, initial_actions=initial_actions)
    episode["reference_protocol"] = args.reference_protocol
    episode["reference_states"] = reference_states
    episode["runtime_joint_limits"] = {k: list(v) for k, v in joint_limits.items()}
    if manifest_path is not None:
        episode["frozen_witness_manifest"] = {"path": str(manifest_path), "sha256": file_sha256(manifest_path)}
    if args.mode == "kimi":
        usage = episode["token_usage"]
        episode["real_api_requests"] = teacher.calls
        episode["usage_cost_rmb_estimate"] = (teacher.pricing.estimate(
            usage["input_tokens"], usage["output_tokens"])
            if teacher.pricing and usage["total_tokens"] else None)
        episode["billing_cost_rmb"] = None
    write_json_atomic(args.out / "episode.json", episode)
    audit_episode(args.out, private, annotation, manifest=manifest)
    return episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "local-qwen", "kimi"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--oracle-annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--sample-id")
    parser.add_argument("--project-fixture", action="store_true",
                        help="Use the project-authored compatibility fixture")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--local-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--playwright-site-packages", type=Path,
                        help="Append an existing Playwright install when the Qwen environment lacks it")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--third-party-transmission-reviewed", action="store_true")
    parser.add_argument("--transmission-review-basis")
    parser.add_argument("--estimated-input-tokens", type=int)
    parser.add_argument("--input-price-rmb", type=float)
    parser.add_argument("--output-price-rmb", type=float)
    parser.add_argument("--pricing-verified-at")
    parser.add_argument("--pricing-source")
    parser.add_argument("--max-cost-rmb", type=float, default=1.0)
    parser.add_argument("--max-real-requests", type=int, default=1)
    parser.add_argument("--decision-attempts", type=int, default=3,
                        help="Bounded schema-correction attempts per decision")
    parser.add_argument("--experiment-mode", choices=("active", "fixed-diagnosis"), default="active")
    parser.add_argument("--reference-protocol", choices=("static_front", REFERENCE_PROTOCOL),
                        default="static_front",
                        help="static_front reproduces round2; front_motion_v1 adds label-independent reference motion states")
    parser.add_argument("--frozen-witnesses", type=Path,
                        help="Directory written by scripts.freeze_teacher_witnesses")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments), indent=2))
