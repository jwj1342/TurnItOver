#!/usr/bin/env python3
"""Summarize completed model/supervision runs under an identical evaluation protocol."""
import argparse
import json
from pathlib import Path


def compare(evaluations):
    if len(evaluations) < 2:
        raise ValueError("At least two labeled evaluations are required")
    reference = next(iter(evaluations.values()))
    required = {"effective_generation_config_sha256", "defect_ids", "processor_sha256", "device", "version", "task", "split", "dataset_version", "dataset_manifest_sha256", "annotations_sha256", "split_sha256",
                "public_input_sha256", "max_new_tokens", "do_sample", "seed", "n", "independent_objects"}
    protocol = reference.get("protocol", {})
    if set(protocol) != required or any(protocol[k] is None for k in required):
        raise ValueError("Incomplete evaluation protocol")
    for label, evaluation in evaluations.items():
        if evaluation.get("status") != "complete":
            raise ValueError(f"Evaluation {label} is incomplete")
        if evaluation.get("protocol") != protocol:
            raise ValueError(f"Controlled comparison protocol mismatch: {label}")
        if evaluation.get("n") != protocol["n"] or evaluation.get("independent_objects") != protocol["independent_objects"]:
            raise ValueError(f"Evaluation denominator mismatch: {label}")
    return {"version": 3, "status": "complete", "protocol": protocol,
            "evaluations": {label: {key: value for key, value in row.items()
                                    if key not in ("protocol", "elapsed_seconds")}
                            for label, row in evaluations.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--sft", type=Path)
    parser.add_argument("--evaluation", action="append", default=[], metavar="LABEL=METRICS_JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"Output must not exist: {args.out}")
    paths = {}
    for label, path in (("base", args.base), ("sft", args.sft)):
        if path:
            paths[label] = path
    for item in args.evaluation:
        label, path = item.split("=", 1)
        if not label or label in paths:
            raise ValueError("Evaluation labels must be unique and nonempty")
        paths[label] = Path(path)
    report = compare({label: json.loads(path.read_text()) for label, path in paths.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
