#!/usr/bin/env python3
"""Four acquisition policies in the existing Chromium loop; private scoring is post-hoc."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_active_qwen import replay_images
from scripts.action_sft_metrics import summarize_rows
from scripts.run_oracle_supervision import _benchmark_provenance
from turnitover.assets.source import make_source
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.actions import Detent
from turnitover.core.serde import to_dict
from turnitover.dataset.prepare import records
from turnitover.models.action_qwen import SharedModel, RemoteModel
from turnitover.models.client import ModelError, ModelResult
from turnitover.oracle.evidence import ExposureThresholds, exposure_matrix, replay_corruptions
from turnitover.oracle.supervision import action_json
from turnitover.oracle.witnesses import Witness, plan, witness_grid
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.verifier.action_only import ACTION_STAGE_INSTRUCTION, ActionOnlyPolicy, complete_final_readout
from turnitover.verifier.contracts import Context
from turnitover.verifier.loop import run_verification


class EngineeringModel:
    """Explicit fake readout for browser plumbing checks; never a scored model result."""
    def call(self, enabled, final=False):
        def generate(prompt, images):
            if not final:
                raise ValueError("Engineering smoke supports Fixed/Random acquisition only")
            return ModelResult(json.dumps({"verdict": {"status": "uncertain", "confidence": 0.,
                "findings": [], "summary": "Engineering smoke", "limitations": ["Fake diagnoser"]}}),
                "engineering-fake", {}, "completed", 0.)
        return generate


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(args):
    if args.out.exists():
        raise ValueError(f"Output must not exist: {args.out}")
    provenance = _benchmark_provenance(args.benchmark)
    cfg = load_generate_config(args.config, repo_root=args.asset_workspace)
    cfg = replace(cfg, render=replace(cfg.render, harness_dir=args.harness_workspace / "web/dist",
                                    esbuild_bin=args.harness_workspace / "web/node_modules/.bin/esbuild"))
    source = make_source(**cfg.asset_source)
    oracle_manifest = args.oracle / "manifest.json"
    if json.loads(oracle_manifest.read_text()).get("status") != "complete":
        raise ValueError("Complete original supervision required")
    # No Test file is opened. Validation IDs, never private difficulty labels, select cases.
    input_path = args.benchmark / "prepared/validation.inputs.jsonl"
    inputs = {row["id"]: row for row in read_jsonl(input_path)}
    raw = {hashlib.sha256(f"{sample.source_id}:{sample.sample_id}".encode()).hexdigest()[:24]: sample
           for sample, _ in records(sorted((args.benchmark / "raw").glob("*.tar")))
           if hashlib.sha256(f"{sample.source_id}:{sample.sample_id}".encode()).hexdigest()[:24] in inputs}
    selected = sorted(inputs)[:args.limit] if args.limit else sorted(inputs)
    args.out.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    public_references = {}
    for row in read_jsonl(args.oracle / "validation.actions.jsonl"):
        name = row["images"][0]
        uid = row["id"].split(":action:")[0]
        digest = hashlib.sha256((args.oracle / name).read_bytes()).hexdigest()
        if row.get("image_sha256") and row["image_sha256"][0] != digest:
            raise ValueError(f"Original reference SHA mismatch: {uid}")
        public_references[uid] = digest
    if not set(selected) <= set(public_references):
        raise ValueError("Validation IDs missing original action reference")
    config.update(frozen_test_used=False, policies=args.modes,
                  action_stage_instruction=ACTION_STAGE_INSTRUCTION,
                  action_stage_instruction_sha256=hashlib.sha256(ACTION_STAGE_INSTRUCTION.encode()).hexdigest(),
                  raw_access="Mixed shards are decoded; only Validation IDs are retained or evaluated",
                  final_diagnoser="base Qwen without adapter, greedy", score_access="after all policy inference",
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [input_path, *provenance, args.config, cfg.views_path,
                                              Path(cfg.asset_source["manifest"]), oracle_manifest,
                                              args.oracle / "validation.actions.jsonl",
                                              *sorted((args.benchmark / "raw").glob("*.tar")), Path(__file__),
                                              ROOT / "turnitover/verifier/action_only.py"]},
                  reference_sha256={uid: public_references[uid] for uid in selected})
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2))
    if args.engineering_smoke and any(mode not in ("fixed", "random") for mode in args.modes):
        raise ValueError("Engineering smoke only accepts fixed/random modes")
    model = (EngineeringModel() if args.engineering_smoke else RemoteModel(args.model_endpoint)
             if args.model_endpoint else SharedModel(args.model, args.adapter, args.action_tokens, args.final_tokens))
    episodes = []
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
        browser = session.browser_version()
        for uid in selected:
            sample = raw[uid]
            reference = ObjectProgram.from_spec(source.get(sample.asset_id))
            candidate = replay_corruptions(reference, sample.corruptions)
            if (reference.sha, candidate.sha) != (sample.reference_sha, sample.program_sha):
                raise ValueError(f"Source identity mismatch: {uid}")
            context = Context(**inputs[uid]["context"])
            joints = {joint.id: joint.limits for joint in reference.spec.joints}
            grid = witness_grid(context.views, joints)
            # Reuse exactly the original teacher's public reference, never regenerate it.
            reference_png = (args.oracle / "images" / uid / "reference.png").read_bytes()
            for budget in args.budgets:
                for mode in config["policies"]:
                    folder = args.out / f"k{budget}" / mode / uid
                    image_dir = folder / "images" / uid
                    image_dir.mkdir(parents=True)
                    reference_path = f"images/{uid}/reference.png"
                    (folder / reference_path).write_bytes(reference_png)
                    # Fixed/random planners receive only public geometry limits and
                    # empty exposures. Their existing behavior ignores exposures.
                    actions = ([action_json(action) for action, _ in plan(
                        grid, [() for _ in grid], joints, budget, mode=mode, seed=sample.seed)]
                               if mode in ("fixed", "random") else None)
                    policy = ActionOnlyPolicy(model.call(mode == "action_sft"), model.call(False, final=True),
                                              folder, reference_path, actions)
                    def sink(step, png):
                        relative = f"images/{uid}/observation-{step:03d}.png"
                        (folder / relative).write_bytes(png)
                        return relative
                    session.load(candidate, framing=sample.framing)
                    try:
                        episode = run_verification(session, policy, context, budget, sink)
                        episode, final_readout = complete_final_readout(policy, context, episode)
                    finally:
                        session.dispose()
                    (folder / "episode.json").write_text(json.dumps(to_dict(episode), indent=2))
                    item = dict(id=uid, asset_id=sample.asset_id, mode=mode, budget=budget,
                                folder=str(folder.relative_to(args.out)), spent=episode.spent,
                                termination=episode.termination, action_calls=len(policy.calls),
                                action_attempts=policy.action_attempts, legal_actions=policy.legal_actions,
                                final_readout=final_readout,
                                action_only_json=sum(call.get("action_only_json", False) for call in policy.calls))
                    episodes.append((item, episode))
                    print(json.dumps(item), flush=True)
        # All learner decisions are completed and saved before private scoring starts.
        scored = []
        for item, episode in episodes:
            sample = raw[item["id"]]
            reference = ObjectProgram.from_spec(source.get(sample.asset_id))
            folder = args.out / item["folder"]
            visual = [observation for observation in episode.trajectory if observation.image_ref]
            if visual:
                candidate_images = [(folder / observation.image_ref).read_bytes() for observation in visual]
                reference_images = replay_images(session, reference, sample.framing, episode.trajectory)
                repaired = [replay_images(session, replay_corruptions(reference, sample.corruptions, index),
                                          sample.framing, episode.trajectory)
                            for index in range(len(sample.corruptions))]
                exposure, signals = exposure_matrix(candidate_images, reference_images, repaired, ExposureThresholds())
                covered = set().union(*map(set, exposure))
            else:
                exposure, signals, covered = [], {}, set()
            expected = {label.defect_id for label in sample.corruptions}
            predicted = {finding.defect_id for finding in episode.verdict.findings}
            item.update(expected=sorted(expected), predicted=sorted(predicted),
                        diagnostic_exact=expected == predicted and episode.verdict.status == ("fail" if expected else "pass"),
                        valid_evidence=bool(covered), all_evidence=len(covered) == len(sample.corruptions),
                        exposed=len(covered), labels=len(sample.corruptions),
                        legal_action_rate=(item["legal_actions"] / item["action_attempts"]
                                           if item["action_attempts"] else None))
            (folder / "evaluation.private.json").write_text(json.dumps(
                dict(**item, exposure=exposure, signals=signals), indent=2))
            scored.append(item)
    summary, paired, objects = summarize_rows(scored)
    oracle_rows = []
    if args.oracle:
        # Read existing privileged annotations only after all learner inference.
        # Reuse the original planner and exposure grid, with each requested K.
        for annotation in read_jsonl(args.oracle / "annotations.private.jsonl"):
            if annotation["split"] != "validation" or annotation["id"] not in selected:
                continue
            sample = raw[annotation["id"]]
            if (annotation["reference_sha"], annotation["candidate_sha"]) != (sample.reference_sha, sample.program_sha):
                raise ValueError("Offline Oracle annotation identity mismatch")
            reference = ObjectProgram.from_spec(source.get(sample.asset_id))
            joints = {joint.id: joint.limits for joint in reference.spec.joints}
            grid = tuple(Witness(row["view"], row["joint"],
                                 Detent(row["detent"]) if row["detent"] else None)
                         for row in annotation["grid"])
            for budget in args.budgets:
                trajectory = plan(grid, annotation["exposures"], joints, budget,
                                  mode="oracle", seed=sample.seed)
                covered = set().union(*(set(annotation["exposures"][index]) for _, index in trajectory))
                oracle_rows.append(dict(id=annotation["id"], asset_id=sample.asset_id, budget=budget,
                                        spent=len(trajectory), exposed=len(covered), valid_evidence=bool(covered)))
    report = dict(status="engineering_smoke" if args.engineering_smoke else "complete", browser=browser,
                  score_eligible=not args.engineering_smoke, frozen_test_used=False, summary=summary,
                  per_object_paired_deltas=paired, offline_privileged_oracle=oracle_rows,
                  per_object=dict(objects), episodes=scored,
                  limitations=["Pixel counterfactual exposure is an offline proxy, not semantic recognition.",
                               "Validation case counts do not equal independent object counts.",
                               "K=2/4 are budget shifts relative to original K=6 supervision."])
    (args.out / "images.sha256.json").write_text(json.dumps(
        {str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
         for path in sorted(args.out.rglob("*.png"))}, indent=2))
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "asset-workspace", "harness-workspace", "model", "adapter", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True,
                        help="Original supervision root: reference PNGs; privileged annotations only post-inference")
    parser.add_argument("--config", type=Path, default=Path("configs/generate_replicacad.yaml"))
    parser.add_argument("--browser-site-packages", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--action-tokens", type=int, default=128)
    parser.add_argument("--final-tokens", type=int, default=1024)
    parser.add_argument("--model-endpoint", help="SSH-forwarded HTTP inference endpoint")
    parser.add_argument("--modes", nargs="+", choices=["zero_shot", "action_sft", "fixed", "random"],
                        default=["zero_shot", "action_sft", "fixed", "random"])
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    if args.browser_site_packages:
        sys.path.append(str(args.browser_site_packages.resolve()))
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
