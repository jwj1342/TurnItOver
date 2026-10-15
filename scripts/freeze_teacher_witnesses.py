"""Freeze motion-view witnesses and the label-independent reference protocol before teacher calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.teacher.witnesses import build_manifest, sample_framing


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data/benchmarks/partnet-mobility-formal-v1/prepared"
DEFAULT_CONFIG = ROOT / "configs/generate_partnet_mobility_supervision.yaml"
DEFAULT_ANNOTATIONS = (ROOT / "data/benchmarks/partnet-mobility-formal-v1/"
                       "oracle-first-failure-v4/annotations.private.jsonl")


def _rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}


def run(args) -> list[dict]:
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Frozen witness directory must be new or empty")
    cfg = load_generate_config(args.config)
    annotations = _rows(args.oracle_annotations)
    test_ids = set(_rows(args.benchmark / "test.gold.jsonl")) | set(_rows(args.benchmark / "test.inputs.jsonl"))
    tables = {split: (_rows(args.benchmark / f"{split}.inputs.jsonl"),
                      _rows(args.benchmark / f"{split}.gold.jsonl")) for split in ("train", "validation")}
    manifests = []
    with ObservationSession(cfg.render, load_views(cfg.views_path)) as session:
        for uid in args.sample_id:
            split = next((name for name, (inputs, gold) in tables.items() if uid in inputs and uid in gold), None)
            if split is None:
                raise ValueError(f"{uid} is not a prepared Train/Validation row")
            public, gold = tables[split][0][uid], tables[split][1][uid]
            framing = sample_framing(args.benchmark.parent / "raw", gold["source_id"], gold["sample_id"])
            manifest = build_manifest(
                session, uid=uid, split=split, public=public, gold=gold, annotation=annotations[uid],
                candidate=ObjectProgram(gold["candidate_program"]),
                reference=ObjectProgram(gold["reference_program"]), framing=framing,
                annotation_root=args.oracle_annotations.parent, out_dir=args.out / uid,
                source_files={"inputs": args.benchmark / f"{split}.inputs.jsonl",
                              "gold": args.benchmark / f"{split}.gold.jsonl",
                              "annotations": args.oracle_annotations},
                test_ids=test_ids, prepared_reference=args.benchmark / public["reference_images"][0],
                supersedes=args.supersedes / uid / "manifest.json" if args.supersedes else None)
            manifests.append({"id": uid, "asset_id": manifest["asset_id"],
                              "objective_defect_ids": manifest["objective_defect_ids"],
                              "witnesses": len(manifest["witnesses"]),
                              "v4_pixel_verified": sum(w["v4_pixel_verified"] for w in manifest["witnesses"]),
                              "reference_states": len(manifest["reference_protocol"]["states"]),
                              "fixed_protocol_steps": len(manifest["fixed_protocol_actions"]),
                              "reference_support": manifest["reference_support"],
                              "perceptible_support": manifest["perceptible_support"],
                              "rest_hierarchy_equal": manifest["rest_hierarchy"]["equal"],
                              "extra_protocol_steps": {name: len(row["fixed_protocol_actions"]) for name, row
                                                       in manifest["extra_reference_protocols"].items()}})
            print(json.dumps(manifests[-1]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(manifests, indent=2))
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--oracle-annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--supersedes", type=Path,
                        help="Earlier frozen witness directory whose manifests must stay field-identical")
    run(parser.parse_args())
