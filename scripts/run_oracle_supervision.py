"""Render counterfactual witnesses, replay budgeted controls, and export causal action BC.

More than 16 samples must run through Slurm. No model training or external APIs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from turnitover.assets.source import make_source
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.dataset.prepare import records, SPLITS
from turnitover.oracle.evidence import ExposureThresholds, exposure_matrix, pixels, render_grid, replay_corruptions
from turnitover.oracle.episodes import episode_examples
from turnitover.oracle.stopping import stopping_examples
from turnitover.oracle.supervision import action_examples, action_json
from turnitover.oracle.witnesses import plan, witness_grid
from turnitover.policy.loop import execute_observation
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views


IDENTITY_KEYS = (
    "formal_training", "catalog_role", "dataset_revision", "source_dataset", "source_asset_id",
    "source_identity_sha256", "source_content_sha256", "spec_sha256", "dataset", "source_identity",
    "annotation_id", "model_id", "archive_sha256",
)


def _attach_identity(example, output, sample, source_metadata, reference, candidate):
    identity = {
        "source_id": sample.source_id, "asset_id": sample.asset_id,
        "split_group": sample.dataset["split_group"],
        "reference_sha": reference.sha, "candidate_sha": candidate.sha,
        **{key: source_metadata[key] for key in IDENTITY_KEYS if key in source_metadata},
    }
    example["data_identity"] = identity
    example["image_sha256"] = [hashlib.sha256((output / path).read_bytes()).hexdigest()
                               for path in example["images"]]
    return example


def _benchmark_provenance(benchmark):
    prepared = benchmark / "prepared/manifest.json"
    prepared_record = json.loads(prepared.read_text())
    if prepared_record.get("status") != "complete":
        raise ValueError("Complete prepared dataset required")
    manifests = sorted((benchmark / "raw").glob("*.manifest.json"))
    if not manifests or any(json.loads(path.read_text()).get("status") != "complete" for path in manifests):
        raise ValueError("Complete raw shards required")
    top = benchmark / "benchmark.json"
    if top.exists() and json.loads(top.read_text()).get("status") != "complete":
        raise ValueError("Complete benchmark required")
    return [prepared, *manifests, *([top] if top.exists() else [])]


def run(benchmark, config, output, budget=6, per_asset=None, episode_objective="first_failure"):
    if budget < 1 or (per_asset is not None and per_asset < 1):
        raise ValueError("Positive budget and per-asset limit required")
    benchmark_files = _benchmark_provenance(benchmark)
    prepared_manifest = json.loads((benchmark / "prepared/manifest.json").read_text())
    cfg = load_generate_config(config)
    source = make_source(**cfg.asset_source)
    inputs = {}
    for split in SPLITS:
        for line in (benchmark / "prepared" / f"{split}.inputs.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["id"] in inputs:
                raise ValueError("Duplicate sample across splits")
            inputs[row["id"]] = (split, row)
    output.mkdir(parents=True, exist_ok=False)
    thresholds = ExposureThresholds()
    report = {"version": 1, "status": "incomplete", "budget": budget,
              "formal_training": prepared_manifest.get("formal_training", False),
              "thresholds": asdict(thresholds), "benchmark": str(benchmark.resolve()),
              "condition": "privileged_counterfactual_pixel_exposure",
              "episode_objective": episode_objective,
              "limitations": ["Pixel exposure is a proxy, not human or VLM recognition.",
                              "Greedy finite-grid oracle is not a global optimal policy.",
                              "Fixed/random controls order witnesses, not arbitrary atomic actions.",
                              "Teacher render costs are excluded from learner action budget.",
                              "Action targets use privileged truth; inputs contain only a static reference and past observations.",
                              "Episode stopping semantics are explicit and included in every model-visible payload.",
                              "No diagnosis SFT or model capability claim is made by this export."]}
    root = Path(__file__).resolve().parents[1]
    provenance_files = [config, cfg.views_path, *sorted((root / "turnitover").rglob("*.py")),
                        Path(__file__), root / "turnitover/verifier/prompt.txt",
                        *benchmark_files]
    report["source_hashes"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in provenance_files}
    report["render_resolution"] = [cfg.render.width, cfg.render.height]
    report["fixed_order"] = "Each view at rest, then each joint's unique nonzero detents for front/oblique views"
    counts, episode_counts, stopping_counts = Counter(), Counter(), Counter()
    episode_terminations, stop_targets, samples = Counter(), Counter(), Counter()
    curves = {mode: np.zeros(budget, dtype=int) for mode in ("oracle", "fixed", "random")}
    total_labels, teacher_renders = 0, 0
    reference_cache = {}
    streams, episode_streams, stopping_streams = {}, {}, {}
    try:
        for split in SPLITS:
            streams[split] = (output / f"{split}.actions.jsonl").open("w")
            episode_streams[split] = (output / f"{split}.episodes.jsonl").open("w")
            stopping_streams[split] = (output / f"{split}.stopping.jsonl").open("w")
        with (output / "annotations.private.jsonl").open("w") as annotations, ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
            for sample, _ in records(sorted((benchmark / "raw").glob("*.tar"))):
                uid = hashlib.sha256(f"{sample.source_id}:{sample.sample_id}".encode()).hexdigest()[:24]
                if uid not in inputs:
                    continue  # prepared quarantine already excluded these samples
                if per_asset is not None and samples[sample.asset_id] >= per_asset:
                    continue
                split, model_input = inputs[uid]
                reference = ObjectProgram.from_spec(source.get(sample.asset_id))
                candidate = replay_corruptions(reference, sample.corruptions)
                if reference.sha != sample.reference_sha or candidate.sha != sample.program_sha:
                    raise ValueError(f"Stale reference or corruption replay mismatch: {uid}")
                source_metadata = source.metadata(sample.asset_id)
                if any(source_metadata.get(key) != sample.dataset[key]
                       for key in IDENTITY_KEYS if key in sample.dataset):
                    raise ValueError(f"Asset provenance changed since generation: {uid}")
                joints = {j.id: j.limits for j in reference.spec.joints}
                grid = witness_grid(model_input["context"]["views"], joints)
                cache_key = (reference.sha, sample.framing)
                if cache_key not in reference_cache:
                    reference_cache[cache_key] = render_grid(session, reference, sample.framing, grid)
                    teacher_renders += len(grid)
                ref_images = reference_cache[cache_key]
                cand_images = render_grid(session, candidate, sample.framing, grid)
                repaired = [render_grid(session, replay_corruptions(reference, sample.corruptions, i), sample.framing, grid)
                            for i in range(len(sample.corruptions))]
                teacher_renders += len(grid)*(1+len(repaired))
                exposures, signals = exposure_matrix(cand_images, ref_images, repaired, thresholds)
                front = next(i for i, w in enumerate(grid) if w.key == "front:rest:rest")
                folder = output / "images" / uid
                folder.mkdir(parents=True)
                reference_path = Path("images") / uid / "reference.png"
                (output / reference_path).write_bytes(ref_images[front])
                trajectories = {}
                for mode in curves:
                    trajectory = plan(grid, exposures, joints, budget, mode=mode, seed=sample.seed)
                    history, covered, curve = [], set(), []
                    session.load(candidate, framing=sample.framing)
                    try:
                        for step, (action, witness_index) in enumerate(trajectory):
                            def sink(index, png):
                                if not np.array_equal(pixels(png), pixels(cand_images[witness_index])):
                                    raise ValueError(f"Replay pixels differ from witness: {uid}/{mode}/{index}")
                                path = Path("images") / uid / f"{mode}-{index:03d}.png"
                                (output / path).write_bytes(png)
                                return path.as_posix()
                            obs = execute_observation(session, action, step, sink)
                            payload = {k: v for k, v in obs.payload.items() if k != "render_ms"}
                            history.append({"step": step, "action": action_json(action), "image_ref": obs.image_ref, "payload": payload})
                            covered.update(exposures[witness_index])
                            curve.append(len(covered))
                    finally:
                        session.dispose()
                    curve.extend([len(covered)]*(budget-len(curve)))
                    curves[mode] += curve
                    trajectories[mode] = {"history": history, "witness_indices": [i for _, i in trajectory], "exposure_curve": curve}
                for example in action_examples(uid, model_input["context"], reference_path.as_posix(), trajectories["oracle"]["history"], budget):
                    _attach_identity(example, output, sample, source_metadata, reference, candidate)
                    streams[split].write(json.dumps(example)+"\n")
                    counts[split] += 1
                oracle = trajectories["oracle"]
                labels = [asdict(c) for c in sample.corruptions]
                episodes = episode_examples(
                        uid, model_input["context"], reference_path.as_posix(), oracle["history"],
                        oracle["witness_indices"], exposures, labels, budget, episode_objective)
                episode_terminations[next(row for row in episodes if row["terminal"])["termination_reason"]] += 1
                for example in episodes:
                    _attach_identity(example, output, sample, source_metadata, reference, candidate)
                    episode_streams[split].write(json.dumps(example)+"\n")
                    episode_counts[split] += 1
                for example in stopping_examples(
                        uid, model_input["context"], reference_path.as_posix(), oracle["history"],
                        oracle["witness_indices"], exposures, labels, budget, episode_objective):
                    _attach_identity(example, output, sample, source_metadata, reference, candidate)
                    stopping_streams[split].write(json.dumps(example)+"\n")
                    stopping_counts[split] += 1
                    stop_targets[f"{example['stop_target']}:{example['stop_outcome'] or 'none'}"] += 1
                annotations.write(json.dumps({"id": uid, "split": split, "asset_id": sample.asset_id,
                    "split_group": sample.dataset["split_group"], "reference_sha": reference.sha,
                    "candidate_sha": candidate.sha, "labels": labels,
                    "data_identity": {key: source_metadata[key] for key in IDENTITY_KEYS if key in source_metadata},
                    "grid": [asdict(w) for w in grid], "exposures": exposures, "signals": signals,
                    "trajectories": trajectories})+"\n")
                samples[sample.asset_id] += 1
                total_labels += len(sample.corruptions)
                print(json.dumps({"sample": uid, "asset": sample.asset_id, "n": sum(samples.values())}), flush=True)
        if not samples:
            raise ValueError("No eligible samples")
        report.update(status="complete", samples=dict(samples), action_examples=dict(counts),
                      episode_examples=dict(episode_counts),
                      stopping_examples=dict(stopping_counts), stop_targets=dict(stop_targets),
                      episode_termination_reasons=dict(episode_terminations),
                      total_labels=total_labels, teacher_renders=teacher_renders,
                      exposed_counts={k: v.tolist() for k, v in curves.items()},
                      exposure_recall={k: (v/total_labels).tolist() if total_labels else None for k, v in curves.items()},
                      replay_validation="Every saved action image equals its independently rendered witness pixel-for-pixel")
    except Exception as exc:
        report.update(status="incomplete", error=str(exc))
        raise
    finally:
        for stream in streams.values():
            stream.close()
        for stream in episode_streams.values():
            stream.close()
        for stream in stopping_streams.values():
            stream.close()
        (output / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmarks/replicacad-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/generate_replicacad.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--per-asset", type=int)
    parser.add_argument("--episode-objective", choices=("first_failure", "comprehensive_within_budget"),
                        default="first_failure")
    args = parser.parse_args()
    print(json.dumps(run(args.benchmark, args.config, args.out, args.budget, args.per_asset,
                         args.episode_objective), indent=2))
