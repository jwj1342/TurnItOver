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
from turnitover.oracle.supervision import action_examples, action_json
from turnitover.oracle.witnesses import plan, witness_grid
from turnitover.policy.loop import execute_observation
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views


def run(benchmark, config, output, budget=6, per_asset=None):
    if budget < 1 or (per_asset is not None and per_asset < 1):
        raise ValueError("Positive budget and per-asset limit required")
    if json.loads((benchmark / "benchmark.json").read_text())["status"] != "complete":
        raise ValueError("Complete benchmark required")
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
              "thresholds": asdict(thresholds), "benchmark": str(benchmark.resolve()),
              "condition": "privileged_counterfactual_pixel_exposure",
              "limitations": ["Pixel exposure is a proxy, not human or VLM recognition.",
                              "Greedy finite-grid oracle is not a global optimal policy.",
                              "Fixed/random controls order witnesses, not arbitrary atomic actions.",
                              "Teacher render costs are excluded from learner action budget.",
                              "Action targets use privileged truth; inputs contain only a static reference and past observations.",
                              "No diagnosis SFT or model capability claim is made by this export."]}
    root = Path(__file__).resolve().parents[1]
    provenance_files = [config, cfg.views_path, *sorted((root / "turnitover").rglob("*.py")),
                        Path(__file__), root / "turnitover/verifier/prompt.txt",
                        benchmark / "splits.json", benchmark / "prepared/manifest.json"]
    report["source_hashes"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in provenance_files}
    report["render_resolution"] = [cfg.render.width, cfg.render.height]
    report["fixed_order"] = "Each view at rest, then each joint's unique nonzero detents for front/oblique views"
    counts, samples = Counter(), Counter()
    curves = {mode: np.zeros(budget, dtype=int) for mode in ("oracle", "fixed", "random")}
    total_labels, teacher_renders = 0, 0
    reference_cache = {}
    streams = {}
    try:
        for split in SPLITS:
            streams[split] = (output / f"{split}.actions.jsonl").open("w")
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
                    streams[split].write(json.dumps(example)+"\n")
                    counts[split] += 1
                annotations.write(json.dumps({"id": uid, "split": split, "asset_id": sample.asset_id,
                    "split_group": sample.dataset["split_group"], "reference_sha": reference.sha,
                    "candidate_sha": candidate.sha, "labels": [asdict(c) for c in sample.corruptions],
                    "grid": [asdict(w) for w in grid], "exposures": exposures, "signals": signals,
                    "trajectories": trajectories})+"\n")
                samples[sample.asset_id] += 1
                total_labels += len(sample.corruptions)
                print(json.dumps({"sample": uid, "asset": sample.asset_id, "n": sum(samples.values())}), flush=True)
        if not samples:
            raise ValueError("No eligible samples")
        report.update(status="complete", samples=dict(samples), action_examples=dict(counts),
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
        (output / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmarks/replicacad-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/generate_replicacad.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--per-asset", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.benchmark, args.config, args.out, args.budget, args.per_asset), indent=2))
