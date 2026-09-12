"""Export paired-state diagnosis examples from a completed, replay-validated oracle run."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from turnitover.assets.source import make_source
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.dataset.prepare import records, SPLITS
from turnitover.oracle.diagnosis import paired_example
from turnitover.oracle.evidence import render_grid
from turnitover.oracle.witnesses import witness_grid
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views


def run(oracle, config, output):
    manifest = json.loads((oracle / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Complete oracle export required")
    benchmark = Path(manifest["benchmark"])
    cfg = load_generate_config(config)
    if manifest["render_resolution"] != [cfg.render.width, cfg.render.height]:
        raise ValueError("Render resolution differs from oracle")
    source = make_source(**cfg.asset_source)
    samples = {hashlib.sha256(f"{s.source_id}:{s.sample_id}".encode()).hexdigest()[:24]: s
               for s, _ in records(sorted((benchmark / "raw").glob("*.tar")))}
    inputs = {}
    for split in SPLITS:
        for line in (benchmark / "prepared" / f"{split}.inputs.jsonl").read_text().splitlines():
            row = json.loads(line)
            inputs[row["id"]] = (split, row["context"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"version": 1, "status": "incomplete", "oracle": str(oracle.resolve()),
              "oracle_manifest_sha256": hashlib.sha256((oracle / "manifest.json").read_bytes()).hexdigest(),
              "condition": "paired_reference_diagnosis",
              "limitations": ["Additional matched references are privileged experimental inputs.",
                              "Targets use pixel exposure as a proxy for recognizable evidence.",
                              "No pass targets: finite observations do not establish clean geometry.",
                              "Severity comes from synthetic truth; confidence is uncalibrated."]}
    streams, cache, counts, statuses = {}, {}, Counter(), Counter()
    try:
        for split in SPLITS:
            streams[split] = (output / f"{split}.diagnosis.jsonl").open("w")
        with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
            for line in (oracle / "annotations.private.jsonl").read_text().splitlines():
                annotation = json.loads(line)
                sample = samples[annotation["id"]]
                split, context = inputs[annotation["id"]]
                if split != annotation["split"] or sample.dataset["split_group"] != annotation["split_group"]:
                    raise ValueError("Split mismatch")
                reference = ObjectProgram.from_spec(source.get(sample.asset_id))
                if reference.sha != annotation["reference_sha"] or sample.program_sha != annotation["candidate_sha"]:
                    raise ValueError("Stale source")
                key = (reference.sha, sample.framing)
                grid = witness_grid(context["views"], {j.id: j.limits for j in reference.spec.joints})
                from dataclasses import asdict
                if [asdict(w) for w in grid] != annotation["grid"]:
                    raise ValueError("Witness grid changed")
                if key not in cache:
                    images = render_grid(session, reference, sample.framing, grid)
                    paths = []
                    for index, png in enumerate(images):
                        path = Path("images") / sample.asset_id / f"reference-{len(cache)}-{index:03d}.png"
                        (output / path).parent.mkdir(parents=True, exist_ok=True)
                        (output / path).write_bytes(png)
                        paths.append(path.as_posix())
                    cache[key] = paths
                for mode, trajectory in annotation["trajectories"].items():
                    candidates = []
                    for observation in trajectory["history"]:
                        path = Path(observation["image_ref"])
                        if path.is_absolute() or ".." in path.parts:
                            raise ValueError("Unsafe observation path")
                        (output / path).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(oracle / path, output / path)
                        candidates.append(path.as_posix())
                    references = [cache[key][index] for index in trajectory["witness_indices"]]
                    row = paired_example(annotation, context, mode, candidates, references)
                    streams[split].write(json.dumps(row)+"\n")
                    counts[split] += 1
                    statuses[json.loads(row["messages"][1]["content"])["verdict"]["status"]] += 1
        report.update(status="complete", counts=dict(counts), statuses=dict(statuses))
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
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/generate_replicacad.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.oracle, args.config, args.out), indent=2))
