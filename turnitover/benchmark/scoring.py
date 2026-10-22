"""Deterministic Benchmark v1 scoring. Reads frozen manifest/gold and a predictions JSONL; no clock, no RNG."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from turnitover.benchmark.manifest import load_gold, load_manifest, verify_freeze
from turnitover.benchmark.protocol import (ABSTAIN, CONDITIONS, LABEL_CLEAN, InvalidOutput, Prediction,
                                           finding_consistent, parse_prediction, target_key)
from turnitover.core.jsonio import write_json_atomic
from turnitover.taxonomy import load_taxonomy

SCORE_SCHEMA_VERSION = 1


def _ratio(num: float, den: float):
    return None if den == 0 else num / den


def _f1(tp: int, fp: int, fn: int):
    return None if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def read_predictions(path: Path) -> dict[str, tuple[dict | None, str | None]]:
    """case_id -> (object, error). Unparseable lines are kept as errors under a synthetic key."""
    rows: dict[str, tuple[dict | None, str | None]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            rows[f"<line {number}>"] = (None, "not valid JSON")
            continue
        case_id = obj.get("case_id") if isinstance(obj, dict) else None
        key = case_id if isinstance(case_id, str) else f"<line {number}>"
        if key in rows:
            raise ValueError(f"duplicate prediction for {key}")
        rows[key] = (obj, None)
    return rows


def _labels(prediction: Prediction | None) -> set[str]:
    if prediction is None or prediction.decision in ABSTAIN:
        return set()
    return {LABEL_CLEAN} if prediction.decision == "pass" else {f.defect_id for f in prediction.findings}


def score(bench_dir: Path, predictions_path: Path, output_path: Path, *, condition: str,
          split: str = "validation") -> dict:
    if output_path.exists():
        raise ValueError("Scoring output must not exist")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    freeze = verify_freeze(bench_dir)
    manifest = {r["sample_id"]: r for r in load_manifest(bench_dir, split)}
    gold = load_gold(bench_dir, split)
    if not manifest:
        raise ValueError(f"no samples in split {split!r}")
    raw = read_predictions(predictions_path)
    unknown = sorted(k for k in raw if k not in manifest and not k.startswith("<line"))
    if unknown:
        raise ValueError(f"predictions for samples outside split {split!r}: {unknown[:3]}")
    taxonomy_ids = set(load_taxonomy().ids)
    sufficient = CONDITIONS[condition]["evidence_sufficient"]

    counts, classes, rows = Counter(), {}, []
    loc = Counter()
    by_difficulty: dict[str, Counter] = {}
    for sample_id in sorted(manifest):
        row, g = manifest[sample_id], gold[sample_id]
        obj, error = raw.get(sample_id, (None, "missing prediction"))
        prediction = None
        if obj is not None:
            try:
                prediction = parse_prediction(obj, condition=condition, taxonomy_ids=taxonomy_ids,
                                              joint_ids=set(row["context"]["joint_ids"]),
                                              part_ids=set(row["context"]["part_ids"]))
            except InvalidOutput as exc:
                error = str(exc)
        valid = prediction is not None
        decision = prediction.decision if valid else "invalid"
        gold_keys = {target_key(d["defect_id"], d["joint_id"], d["parts"]) for d in g["defects"]}
        pred_keys = {f.key for f in prediction.findings} if valid else set()
        is_defect = g["status"] == "fail"
        abstain_expected = g["status"] == "uncertain" or not sufficient
        if abstain_expected:
            exact = decision in ABSTAIN
        else:
            exact = valid and decision == g["status"] and pred_keys == gold_keys

        gold_labels = {d["defect_id"] for d in g["defects"]} or ({LABEL_CLEAN} if g["status"] == "pass" else set())
        pred_labels = _labels(prediction)
        for label in gold_labels | pred_labels:
            c = classes.setdefault(label, Counter())
            c["tp" if label in gold_labels & pred_labels else "fp" if label in pred_labels else "fn"] += 1
        loc["tp"] += len(gold_keys & pred_keys)
        loc["fp"] += len(pred_keys - gold_keys)
        loc["fn"] += len(gold_keys - pred_keys)

        consistent = None
        if valid and decision == "fail":
            consistent = all(finding_consistent(f, condition) for f in prediction.findings)
        local = Counter(n=1, invalid=not valid, exact=exact, defect=is_defect, clean=g["status"] == "pass",
                        false_pass=is_defect and decision == "pass",
                        clean_false_fail=g["status"] == "pass" and decision == "fail",
                        abstain=decision in ABSTAIN, fail_predictions=decision == "fail",
                        consistent=bool(consistent), abstain_expected=abstain_expected,
                        abstain_correct=abstain_expected and decision in ABSTAIN)
        counts.update(local)
        by_difficulty.setdefault(g["difficulty"], Counter()).update(local)
        rows.append({"sample_id": sample_id, "difficulty": g["difficulty"], "decision": decision, "exact": exact,
                     "error": error if not valid else None, "evidence_consistent": consistent})

    n = counts["n"]
    per_class = {k: {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "f1": _f1(c["tp"], c["fp"], c["fn"])}
                 for k, c in sorted(classes.items())}
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    report = {
        "schema_version": SCORE_SCHEMA_VERSION, "benchmark": freeze["benchmark"], "split": split,
        "condition": condition, "freeze_files": freeze["files"], "n": n,
        "gold_defect_samples": counts["defect"], "gold_clean_samples": counts["clean"],
        "metrics": {
            "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "localization_f1": _f1(loc["tp"], loc["fp"], loc["fn"]),
            "exact_verdict": _ratio(counts["exact"], n),
            "evidence_consistency": _ratio(counts["consistent"], counts["fail_predictions"]),
            "false_pass": _ratio(counts["false_pass"], counts["defect"]),
            "invalid_output": _ratio(counts["invalid"], n)},
        "supporting": {
            "uncertain_or_continue_rate": _ratio(counts["abstain"], n),
            "clean_false_fail_rate": _ratio(counts["clean_false_fail"], counts["clean"]),
            "abstain_correct_rate": _ratio(counts["abstain_correct"], counts["abstain_expected"]),
            "fail_predictions": counts["fail_predictions"]},
        "per_class": per_class,
        "per_difficulty": {d: {"n": c["n"], "exact_verdict": _ratio(c["exact"], c["n"]),
                               "false_pass": _ratio(c["false_pass"], c["defect"]),
                               "invalid_output": _ratio(c["invalid"], c["n"])}
                           for d, c in sorted(by_difficulty.items())},
        "per_case": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, report)
    return report
