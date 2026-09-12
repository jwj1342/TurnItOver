"""Build the small mesh benchmark after a matching independent conversion audit.

For >16 samples, submit this through Slurm. No model APIs or training are invoked.
"""
import argparse
import dataclasses
import json
from pathlib import Path

from turnitover.assets.source import make_source
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.dataset.evaluate import evaluate_runtime
from turnitover.dataset.prepare import make_splits, prepare
from turnitover.engine.generate import run_generate


def run(config, audit_path, output, n_samples=None):
    cfg = load_generate_config(config)
    if n_samples is not None:
        cfg = dataclasses.replace(cfg, n_samples=n_samples)
    audit = json.loads(audit_path.read_text())
    verified = {r["asset_id"]: r for r in audit["results"]}
    if audit["status"] != "complete" or not audit.get("passed"):
        raise ValueError("A passing URDF fidelity audit is required")
    for spec in make_source(**cfg.asset_source):
        record = verified.get(spec.asset_id)
        if record is None or not record["passed"] or record.get("program_sha") != ObjectProgram.from_spec(spec).sha:
            raise ValueError(f"Missing/stale conversion audit for {spec.asset_id}")
    if output.exists():
        raise ValueError("Benchmark output must not exist")
    output.mkdir(parents=True)
    manifest = {"status": "incomplete", "config": str(config), "audit": str(audit_path)}
    try:
        shard = run_generate(dataclasses.replace(cfg, output_dir=output / "raw"))
        splits = output / "splits.json"
        make_splits([shard], splits, seed=42)
        prepared = output / "prepared"
        report = prepare([shard], splits, prepared)
        if any(not report["counts"].get(split) for split in ("train", "validation", "test")):
            raise ValueError("Every split must contain accepted samples")
        metrics = evaluate_runtime(prepared, "test", output / "runtime-test")
        manifest.update(status="complete", counts=report["counts"], quality_counts=report["quality_counts"],
                        runtime_metrics=metrics, limitations=["Six asset families only; no trained/active VLM evaluation yet.",
                                                              "Gold targets still require observation-exposure supervision."])
    finally:
        (output / "benchmark.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/generate_replicacad.yaml"))
    parser.add_argument("--audit", type=Path, default=Path("output/replicacad-fk-v3/audit.json"))
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/replicacad-v1"))
    parser.add_argument("--n-samples", type=int)
    args = parser.parse_args()
    run(args.config, args.audit, args.out, args.n_samples)
