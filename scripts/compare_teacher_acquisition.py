"""Execute fixed/random first observations for teacher Episodes without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from turnitover.config import load_generate_config
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.policy.loop import execute_observation
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.verifier.contracts import Context, parse_decision
from turnitover.verifier.policies import VisionPolicy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/benchmarks/partnet-mobility-formal-v1/prepared"
DEFAULT_CONFIG = ROOT / "configs/generate_partnet_mobility_supervision.yaml"


def _rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}


def compare(benchmark: Path, config: Path, sample_ids: list[str], output: Path, seed: int) -> dict:
    if output.exists():
        raise ValueError(f"Output must not exist: {output}")
    inputs = _rows(benchmark / "train.inputs.jsonl")
    gold = _rows(benchmark / "train.gold.jsonl")
    views = load_views(load_generate_config(config).views_path)
    output.mkdir(parents=True)
    rows = []
    for sample_id in sample_ids:
        public, private = inputs[sample_id], gold[sample_id]
        if private.get("provenance", {}).get("formal_training") is not True:
            raise ValueError("Baseline comparison requires formal Train samples")
        context = Context(**public["context"])
        reference_hashes = {
            hashlib.sha256((benchmark / relative).read_bytes()).hexdigest()
            for relative in public["reference_images"]
        }
        for mode in ("fixed", "random"):
            case_dir = output / sample_id / mode
            case_dir.mkdir(parents=True)
            policy = VisionPolicy(None, case_dir, [], mode=mode, seed=seed)
            raw = policy.decide(context, (), 1)
            action = parse_decision(raw, context, set())

            def sink(step: int, png: bytes) -> str:
                path = case_dir / f"observation-{step:03d}.png"
                path.write_bytes(png)
                return str(path.relative_to(output))

            with ObservationSession(load_generate_config(config).render, views) as session:
                session.load(ObjectProgram(private["candidate_program"]), framing=None)
                observation = execute_observation(session, action, 0, sink)
            image_sha = (hashlib.sha256((output / observation.image_ref).read_bytes()).hexdigest()
                         if observation.image_ref else None)
            rows.append({
                "episode_id": sample_id,
                "asset_id": private["asset_id"],
                "split_group": private["split_group"],
                "mode": mode,
                "seed": seed if mode == "random" else None,
                "budget": 1,
                "action": raw["action"],
                "image_ref": observation.image_ref,
                "image_sha256": image_sha,
                "observation_new_information": (
                    image_sha not in reference_hashes if image_sha else bool(observation.payload)),
                "payload": observation.payload,
            })
    result = {
        "schema_version": 1,
        "scope": "same-runner fixed/random acquisition only; no model verdict",
        "model_calls": 0,
        "observation_budget": 1,
        "framing": "candidate auto-framing, matching run_interactive_teacher.py",
        "rows": rows,
    }
    write_json_atomic(output / "comparison.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.benchmark, args.config, args.sample_id, args.out, args.seed), indent=2))


if __name__ == "__main__":
    main()
