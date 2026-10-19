#!/usr/bin/env python3
"""Four acquisition policies in the existing Chromium loop; private scoring is post-hoc."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_active_qwen import replay_images
from scripts.run_oracle_supervision import _benchmark_provenance
from turnitover.assets.source import make_source
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.actions import Detent
from turnitover.core.serde import to_dict
from turnitover.dataset.prepare import records
from turnitover.models.local_qwen import LocalQwen
from turnitover.oracle.evidence import ExposureThresholds, exposure_matrix, replay_corruptions
from turnitover.oracle.supervision import action_json
from turnitover.oracle.witnesses import Witness, plan, witness_grid
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.verifier.action_only import ActionOnlyPolicy
from turnitover.verifier.contracts import Context
from turnitover.verifier.loop import run_verification


class SharedModel:
    """One loaded model, with the action adapter disabled for every final diagnosis."""
    def __init__(self, model_path, adapter, action_tokens, final_tokens):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        self.local = LocalQwen(model_path)
        self.local.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.local.model = PeftModel.from_pretrained(base, adapter, local_files_only=True).eval()
        self.action_tokens, self.final_tokens = action_tokens, final_tokens

    def call(self, enabled, final=False):
        def generate(prompt, images):
            self.local.max_tokens = self.final_tokens if final else self.action_tokens
            with nullcontext() if enabled else self.local.model.disable_adapter():
                return self.local(prompt, images)
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
    # No Test file is opened. Validation IDs, never private difficulty labels, select cases.
    input_path = args.benchmark / "prepared/validation.inputs.jsonl"
    inputs = {row["id"]: row for row in read_jsonl(input_path)}
    raw = {hashlib.sha256(f"{sample.source_id}:{sample.sample_id}".encode()).hexdigest()[:24]: sample
           for sample, _ in records(sorted((args.benchmark / "raw").glob("*.tar")))
           if hashlib.sha256(f"{sample.source_id}:{sample.sample_id}".encode()).hexdigest()[:24] in inputs}
    selected = sorted(inputs)[:args.limit] if args.limit else sorted(inputs)
    args.out.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(frozen_test_used=False, policies=["zero_shot", "action_sft", "fixed", "random"],
                  raw_access="Mixed shards are decoded; only Validation IDs are retained or evaluated",
                  final_diagnoser="base Qwen without adapter, greedy", score_access="after all policy inference",
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [input_path, *provenance, Path(__file__),
                                              ROOT / "turnitover/verifier/action_only.py"]})
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2))
    model = SharedModel(args.model, args.adapter, args.action_tokens, args.final_tokens)
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
            session.load(reference, framing=sample.framing)
            reference_png = session.request_view("front").png
            session.dispose()
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
                    finally:
                        session.dispose()
                    (folder / "episode.json").write_text(json.dumps(to_dict(episode), indent=2))
                    item = dict(id=uid, asset_id=sample.asset_id, mode=mode, budget=budget,
                                folder=str(folder.relative_to(args.out)), spent=episode.spent,
                                termination=episode.termination, action_calls=len(policy.calls),
                                action_attempts=policy.action_attempts, legal_actions=policy.legal_actions,
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
    groups = defaultdict(list)
    objects = defaultdict(list)
    for item in scored:
        groups[f"k{item['budget']}/{item['mode']}"].append(item)
        objects[item["asset_id"]].append(item)
    summary = {key: dict(episodes=len(rows), independent_objects=len({r["asset_id"] for r in rows}),
                         valid_evidence_rate=sum(r["valid_evidence"] for r in rows) / len(rows),
                         all_evidence_rate=sum(r["all_evidence"] for r in rows) / len(rows),
                         diagnosis_accuracy=sum(r["diagnostic_exact"] for r in rows) / len(rows),
                         mean_observation_cost=sum(r["spent"] for r in rows) / len(rows),
                         legal_action_rate=sum(r["legal_actions"] for r in rows) /
                         max(1, sum(r["action_attempts"] for r in rows)))
               for key, rows in groups.items()}
    paired = []
    for asset, rows in objects.items():
        for budget in args.budgets:
            subset = [row for row in rows if row["budget"] == budget]
            averages = {mode: {metric: sum(row[metric] for row in subset if row["mode"] == mode) /
                               sum(row["mode"] == mode for row in subset)
                               for metric in ("valid_evidence", "all_evidence", "diagnostic_exact", "spent")}
                        for mode in config["policies"]}
            for baseline in ("fixed", "random", "zero_shot"):
                paired.append(dict(asset_id=asset, budget=budget, baseline=baseline,
                                   delta_action_sft_minus_baseline={metric: averages["action_sft"][metric] -
                                       averages[baseline][metric] for metric in averages[baseline]}))
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
    report = dict(status="complete", browser=browser, frozen_test_used=False, summary=summary,
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
    parser.add_argument("--oracle", type=Path, help="Existing supervision root, for offline Oracle grid scoring only")
    parser.add_argument("--config", type=Path, default=Path("configs/generate_replicacad.yaml"))
    parser.add_argument("--browser-site-packages", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--action-tokens", type=int, default=128)
    parser.add_argument("--final-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.browser_site_packages:
        sys.path.append(str(args.browser_site_packages.resolve()))
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
