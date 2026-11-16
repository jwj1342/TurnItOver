"""Run Table 2 with the pinned v1.1 evaluator and projected validation inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from table2_projection import project

HASH = "b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "model", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--condition", choices=("viewpoint", "motion", "full"), required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.runtime.resolve() / "scripts"))
    import evaluate_qwen_episode_sft as evaluator

    original_loader = evaluator.load_training_examples
    if args.condition != "full":
        def load_subset(*parameters):
            for record, images in original_loader(*parameters):
                example = project(record, images, args.condition)
                if example is not None:
                    yield example
        evaluator.load_training_examples = load_subset

    result = evaluator.evaluate(
        args.dataset, args.model, args.out, adapter=args.adapter, device="cuda",
        task="diagnosis", split="validation", seed=20261013, max_new_tokens=512,
        dataset_manifest_sha256=HASH,
    )
    if args.condition != "full":
        # The truncated inputs are qualified for findings evaluation only.
        (args.out / "metrics.json").rename(args.out / "internal-evaluator-metrics.json")
    keys = ("n", "independent_objects", "defect_macro_f1", "defect_labels", "defect_per_class")
    summary = {key: result[key] for key in keys}
    summary["by_difficulty"] = {
        tier: {key: row[key] for key in keys}
        for tier, row in result["by_difficulty"].items()
    }
    summary.update(condition=args.condition, status="complete",
                   protocol=result["protocol"],
                   projection_sha256=hashlib.sha256(
                       Path(__file__).with_name("table2_projection.py").read_bytes()).hexdigest())
    (args.out / "table2-f1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
