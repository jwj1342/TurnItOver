"""Evaluate empirical teacher-action learning on Train/Validation, without opening Test."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turnitover.models.action_qwen import SharedModel
from turnitover.oracle.supervision import load_action_groups
from turnitover.oracle.training import tokenize_example, weighted_assistant_loss
from turnitover.verifier.contracts import Context, parse_decision, parse_json


def target_mass(text, group):
    """A supported action earns its empirical probability, including conflicting groups."""
    record = group[0][0]
    payload = json.loads(record["messages"][0]["content"].rsplit("\n", 1)[1])
    value = parse_json(text)
    if not isinstance(value, dict) or set(value) != {"action"}:
        raise ValueError("Action-only output required")
    parse_decision(value, Context(**payload["context"]), {o["step"] for o in payload["history"]})
    return sum(row["loss_weight"] for row, _ in group
               if json.loads(row["messages"][1]["content"]) == value)


def evaluate(args):
    import torch
    if args.out.exists():
        raise ValueError("Fresh output directory required")
    groups = {split: list(load_action_groups(args.dataset, split)) for split in ("train", "validation")}
    model = SharedModel(args.model, args.adapter, args.action_tokens, 1024)
    args.out.mkdir(parents=True)
    summary = {}
    for split, split_groups in groups.items():
        for enabled in (False, True):
            mode = "action_sft" if enabled else "zero_shot"
            rows = []
            for index, group in enumerate(split_groups):
                weighted_nll = 0.
                with torch.inference_mode(), nullcontext() if enabled else model.local.model.disable_adapter():
                    for record, images in group:
                        batch = tokenize_example(model.local.processor, record, images).to(model.local.model.device)
                        output = model.local.model(**batch)
                        weighted_nll += record["loss_weight"] * float(
                            weighted_assistant_loss(output.logits, batch["labels"], [1.]).cpu())
                        del batch, output
                record = group[0][0]
                result = model.call(enabled)(record["messages"][0]["content"],
                                             [args.dataset / path for path in record["images"]])
                error, mass = None, 0.
                try:
                    if result.finish_reason != "completed":
                        raise ValueError("Incomplete generation")
                    mass = target_mass(result.text, group)
                except (ValueError, TypeError, KeyError) as exc:
                    error = str(exc)
                row = dict(group=record["id"].rsplit(":", 1)[0], split=split, mode=mode,
                           weighted_assistant_nll=weighted_nll, legal=error is None,
                           empirical_target_mass=mass, supported_action=mass > 0,
                           majority_mass=max(r["loss_weight"] for r, _ in group),
                           error=error, generation=asdict(result))
                rows.append(row)
                with (args.out / "groups.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(dict(split=split, mode=mode, group=index, nll=weighted_nll,
                                      legal=row["legal"], mass=mass)), flush=True)
            summary[f"{split}/{mode}"] = {metric: sum(row[metric] for row in rows) / len(rows)
                for metric in ("weighted_assistant_nll", "legal", "empirical_target_mass", "supported_action", "majority_mass")}
            summary[f"{split}/{mode}"]["groups"] = len(rows)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(status="complete", frozen_test_used=False, summary=summary,
                  config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  sha256={str(path): digest(path) for path in [args.dataset / "manifest.json",
                      args.dataset / "train.actions.jsonl", args.dataset / "validation.actions.jsonl",
                      args.adapter / "adapter_model.safetensors", Path(__file__)]},
                  semantics="Equal group mean; full probability distribution retained. Teacher empirical mass is not optimal-policy accuracy.")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "model", "adapter", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--action-tokens", type=int, default=128)
    evaluate(parser.parse_args())
