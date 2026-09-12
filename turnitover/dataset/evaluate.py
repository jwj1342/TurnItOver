"""Offline baseline evaluation. Gold is consumed only after a policy decision."""
import json
from collections import Counter
from pathlib import Path

from turnitover.core.actions import Observation
from turnitover.core.serde import from_dict
from turnitover.verifier.contracts import Context, parse_decision
from turnitover.verifier.policies import RuntimePolicy


def evaluate_runtime(dataset: Path, split: str, output: Path):
    if json.loads((dataset / "manifest.json").read_text()).get("status") != "complete":
        raise ValueError("Cannot evaluate an incomplete dataset")
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    gold = {r["id"]: r for r in _rows(dataset / f"{split}.gold.jsonl")}
    if not gold:
        raise ValueError(f"No accepted samples in {split}")
    if output.exists():
        raise ValueError("Evaluation output must not exist")
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps({"status": "incomplete", "policy": "runtime", "split": split}))
    statuses, per_class = Counter(), {}
    seen = set()
    with (output / "predictions.jsonl").open("w") as f:
        for row in _rows(dataset / f"{split}.inputs.jsonl"):
            if row["id"] in seen or row["id"] not in gold:
                raise ValueError("Duplicate input or missing gold")
            seen.add(row["id"])
            history = tuple(from_dict(Observation, {**o, "elapsed_ms": 0}) for o in row["history"])
            context = from_dict(Context, row["context"])
            decision = RuntimePolicy().decide(context, history, 0)
            verdict = parse_decision(decision, context, {o.step for o in history}, final_only=True)
            statuses[verdict.status] += 1
            predicted = {v.defect_id for v in verdict.findings}
            expected = {v["defect_id"] for v in gold[row["id"]]["target"]["findings"]}
            for label in predicted | expected:
                counts = per_class.setdefault(label, Counter())
                counts["tp" if label in predicted & expected else "fp" if label in predicted else "fn"] += 1
            f.write(json.dumps({"id": row["id"], **decision}) + "\n")
    if seen != set(gold):
        raise ValueError("Missing input rows")
    for label, counts in list(per_class.items()):
        tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
        per_class[label] = {"tp": tp, "fp": fp, "fn": fn,
                            "precision": tp/(tp+fp) if tp+fp else None,
                            "recall": tp/(tp+fn) if tp+fn else None,
                            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    result = {"status": "complete", "policy": "runtime", "split": split, "n": len(seen),
              "statuses": dict(statuses), "uncertain_rate": statuses["uncertain"]/len(seen),
              "per_class": per_class, "scope": "detection_on_saved_scripted_observations",
              "limitations": ["Uncertain samples count as missed detections for their gold defects.",
                              "No visual judge, active policy, or generator repair loop was evaluated."]}
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def _rows(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
