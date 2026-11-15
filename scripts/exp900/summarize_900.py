#!/usr/bin/env python3
"""Summarise three-tier-900 runs under out/level900-*/ into one markdown table.

Usage: python summarize_900.py <out_dir> [validation|test]   (default validation)

Columns: defect Macro-F1 per tier, part-localization micro-F1, status accuracy, exact-verdict
accuracy and false-pass count over the full 120-case validation split. SFT rows aggregate
seeds as mean [min, max]. Base rows are single runs. The pixel baseline reports status
accuracy only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def pct(x):
    return None if x is None else 100 * x


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{sum(values) / len(values):.1f} [{min(values):.1f}, {max(values):.1f}]"


def _f1(tp, fp, fn):
    return None if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def supported_macro(pred_path: Path, labels):
    """Per tier: macro-F1 over the labels that have at least one expected finding in that tier
    (her scorer averages over all taxonomy labels, so a tier with 3 of 5 labels caps at 60),
    and type-level micro-F1 (defect label sets per case). Computed from predictions.jsonl."""
    rows = [json.loads(l) for l in pred_path.read_text().splitlines()]
    out = {}
    for tier in ("easy", "medium", "hard"):
        sel = [r for r in rows if r.get("difficulty") == tier]
        if not sel:
            continue
        exp = [{f[0] for f in r["expected_findings"]} for r in sel]
        prd = [{f[0] for f in r["predicted_findings"]} for r in sel]
        support = {lab for e in exp for lab in e}
        per = []
        for lab in labels:
            if lab not in support:
                continue
            tp = sum(lab in e and lab in p for e, p in zip(exp, prd))
            fp = sum(lab in p and lab not in e for e, p in zip(exp, prd))
            fn = sum(lab in e and lab not in p for e, p in zip(exp, prd))
            per.append(_f1(tp, fp, fn) or 0.0)
        tp = sum(len(e & p) for e, p in zip(exp, prd)); fp = sum(len(p - e) for e, p in zip(exp, prd)); fn = sum(len(e - p) for e, p in zip(exp, prd))
        out[tier] = {"supported_macro": pct(sum(per) / len(per)) if per else None, "type_micro": pct(_f1(tp, fp, fn) or 0.0), "n_labels": len(per)}
    return out


def read_metrics(path: Path):
    report = json.loads(path.read_text())
    if report.get("status") != "complete":
        return None
    by = report["by_difficulty"]
    extra = supported_macro(path.with_name("predictions.jsonl"), report["protocol"]["defect_ids"])
    return {"easy": pct(by.get("easy", {}).get("defect_macro_f1")),
            "easy_s": extra.get("easy", {}).get("supported_macro"), "medium_s": extra.get("medium", {}).get("supported_macro"), "hard_s": extra.get("hard", {}).get("supported_macro"),
            "easy_m": extra.get("easy", {}).get("type_micro"), "medium_m": extra.get("medium", {}).get("type_micro"), "hard_m": extra.get("hard", {}).get("type_micro"),
            "medium": pct(by.get("medium", {}).get("defect_macro_f1")),
            "hard": pct(by.get("hard", {}).get("defect_macro_f1")),
            "loc": pct(report["part_localization_micro"]["f1"]),
            "status": pct(report["status_accuracy"]),
            "exact": pct(report["exact_verdict_accuracy"]),
            "false_pass": report["false_pass_count"],
            "support": pct(report["evidence_support_rate"]), "n_findings": report["predicted_findings"],
            "finding_micro": pct(report["finding_micro"]["f1"]),
            "valid": pct(report["valid_output_rate"]), "n": report["n"]}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"   # validation | test
    groups = defaultdict(list)
    for path in sorted(root.glob(f"level900-*/*/base-{split}/metrics.json")):
        m = read_metrics(path)
        if m:
            groups[(path.parts[-3], "Base, zero-shot")].append(m)
    latest = {}
    for path in sorted(root.glob(f"level900-*/*/seed*/*-diagnosis/eval-{split}/metrics.json"), key=lambda p: int(p.parts[-6].rsplit("-", 1)[-1])):
        m = read_metrics(path)
        if m:
            sup = path.parts[-3].split("-")[0]
            latest[(path.parts[-5], f"{sup.capitalize()} diagnosis SFT", path.parts[-4])] = m   # one run per seed, latest job wins
    for (size, training, _seed), m in latest.items():
        groups[(size, training)].append(m)
    print("Table A. Her scorer: per-tier defect Macro-F1 over all 5 taxonomy labels (each validation tier has 3 labels with support, so the ceiling is 60), part-localization micro-F1, status accuracy, exact verdict, false pass.")
    print("| model | training | runs | Easy F1 | Medium F1 | Hard F1 | Loc F1 | status acc | exact verdict | false pass | valid % |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (size, training), ms in sorted(groups.items()):
        cells = [fmt([m[k] for m in ms]) for k in ("easy", "medium", "hard", "loc", "status", "exact")]
        fp = fmt([float(m["false_pass"]) for m in ms])
        print(f"| Qwen3-VL-{size} | {training} | {len(ms)} | " + " | ".join(cells) + f" | {fp} | {fmt([m['valid'] for m in ms])} |")
    for path in sorted(root.glob("level900-*/pixel-status/metrics.json"))[:1]:   # identical in every base job; print once
        r = json.loads(path.read_text())
        v = r["validation"]; bd = r["validation_by_difficulty"]
        tiers = ", ".join(f"{d} {100 * bd[d]['status_accuracy']:.1f}" for d in ("easy", "medium", "hard"))
        print(f"| pixel difference | status only, threshold fit on train | 1 | -- | -- | -- | -- | {100 * v['status_accuracy']:.1f} (by tier {tiers}) | -- | fail predicted uncertain {v['fail_predicted_uncertain']}/{v['fail_cases']} | -- |")
        print(f"| trivial | always fail / always uncertain | -- | -- | -- | -- | -- | {100 * v['always_fail_status_accuracy']:.1f} / {100 * v['always_uncertain_status_accuracy']:.1f} | -- | -- | -- |")

    print()
    print("Table B. Same predictions, per-tier defect-type F1 restricted to labels present in that tier: macro over supported labels / type-level micro-F1.")
    print("| model | training | runs | Easy macro / micro | Medium macro / micro | Hard macro / micro |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for (size, training), ms in sorted(groups.items()):
        cells = [f"{fmt([m[t + '_s'] for m in ms])} / {fmt([m[t + '_m'] for m in ms])}" for t in ("easy", "medium", "hard")]
        print(f"| Qwen3-VL-{size} | {training} | {len(ms)} | " + " | ".join(cells) + " |")
    print()
    print("Table C. Grounding: predicted findings over 120 cases, share of predicted findings whose cited evidence steps match the audited exposing steps (evidence support rate), and full-signature finding micro-F1 (defect, joint, parts).")
    print("| model | training | runs | findings | evidence support % | finding micro-F1 |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for (size, training), ms in sorted(groups.items()):
        print(f"| Qwen3-VL-{size} | {training} | {len(ms)} | {fmt([float(m['n_findings']) for m in ms])} | {fmt([m['support'] for m in ms])} | {fmt([m['finding_micro'] for m in ms])} |")


if __name__ == "__main__":
    main()
