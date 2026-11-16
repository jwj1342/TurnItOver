"""Recognition numbers for tables/recognition.tex and tables/recognition_sizes.tex, with explicit run selection.

Every (size, supervision, seed) and every zero-shot base contributes exactly one evaluation, chosen by hardware:
  --hardware main   H100 (Nibi) when it exists, else L40S (Vulcan): the paper's main rows.
  --hardware l40s   only L40S evaluations (same-hardware comparison with the withheld-evidence runs).
Hardware comes from the first line of <run>/env.txt. Two evaluations of the same item on the same hardware are an error.
Columns: per-tier and overall defect Macro-F1 (her scorer), part-localization F1, evidence support, findings, and from
predictions + export: Unexp. = share of findings whose defect is not an exposed gold label of the case, None exp. = share
of findings on cases without any exposed gold label (as scripts/exp900/analysis/unexposed_rate.py defines them).
Data version (third argument): v1 (default; runs level900-*, export three-tier-900-sft-v1) or v11 (runs level900v11-*,
three-tier-900-sft-v1.1-pre.1, whose evaluator already drops the 16 quarantined validation cases).
Usage: python scripts/exp900/analysis/paper_recognition.py [validation|test] [main|l40s] [v1|v11]
  -> prints and writes $TIO_RUNS/out/RESULTS-recognition-paper-<split>-<hardware>[-v11].md
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

TIO = Path(os.environ.get("TIO_RUNS", Path(__file__).resolve().parents[3] / "tio-runs"))
OUT = TIO / "out"
DATA = sys.argv[3] if len(sys.argv) > 3 else "v1"
EXPORT = TIO / ("data900/v11-pre1-download/three-tier-900-sft-v1.1" if DATA == "v11" else "data900/three-tier-900-sft-v1/episode_export")
PREFIX = "level900v11-" if DATA == "v11" else "level900-"
SUPS = ("zero-shot", "ordinary", "evidence")


def hardware(run: Path) -> str:
    line = (run / "env.txt").read_text().splitlines()[0]
    return "l40s" if "L40S" in line else "h100"


def exposure(split: str) -> tuple[dict, dict]:
    gold, exposed = {}, {}
    labels = {json.loads(l)["id"]: {x["defect_id"] for x in json.loads(l)["labels"]} for l in (EXPORT / "annotations.private.jsonl").open()}
    for line in (EXPORT / f"{split}.episodes.jsonl").open():
        ep = json.loads(line)
        cid = ep["episode_id"]
        target = json.loads(ep["messages"][-1]["content"])["verdict"]
        gold[cid] = labels[cid]
        exposed[cid] = {f["defect_id"] for f in target["findings"] if f["evidence_steps"]}
    return gold, exposed


def candidates(split: str):
    for path in OUT.glob(f"{PREFIX}*/*/seed*/*-diagnosis/eval-{split}/metrics.json"):
        run = path.parents[4]
        yield (path.parents[3].name, path.parents[1].name.split("-")[0], path.parents[2].name[4:]), run, path
    for path in OUT.glob(f"{PREFIX}*/*/base-{split}/metrics.json"):
        yield (path.parents[1].name, "zero-shot", "-"), path.parents[2], path


def select(split: str, mode: str) -> dict:
    found = defaultdict(dict)
    for key, run, path in candidates(split):
        hw = hardware(run)
        if hw in found[key]:
            raise SystemExit(f"two {hw} evaluations of {key}: {found[key][hw][0].name} and {run.name}")
        found[key][hw] = (run, path)
    chosen = {}
    for key, by_hw in found.items():
        if mode == "l40s":
            if "l40s" in by_hw:
                chosen[key] = (*by_hw["l40s"], "l40s")
        else:
            hw = "h100" if "h100" in by_hw else "l40s"
            chosen[key] = (*by_hw[hw], hw)
    return chosen


def metrics(path: Path, gold: dict, exposed: dict) -> dict:
    m = json.loads(path.read_text())
    assert m["status"] == "complete", path
    by = m["by_difficulty"]
    n_find = unexp = none_exp = 0
    for line in path.with_name("predictions.jsonl").open():
        r = json.loads(line)
        ok = gold[r["episode_id"]] & exposed[r["episode_id"]]
        findings = r.get("predicted_findings") or []
        n_find += len(findings)
        unexp += sum(f[0] not in ok for f in findings)
        none_exp += len(findings) if not ok else 0
    pct = lambda v: None if v is None else 100 * v
    return {"easy": pct(by["easy"]["defect_macro_f1"]), "medium": pct(by["medium"]["defect_macro_f1"]),
            "hard": pct(by["hard"]["defect_macro_f1"]), "all": pct(m["defect_macro_f1"]),
            "loc": pct(m["part_localization_micro"]["f1"]), "support": pct(m["evidence_support_rate"]) if n_find else None,
            "findings": float(m["predicted_findings"]), "unexp": 100 * unexp / n_find if n_find else None,
            "none_exp": 100 * none_exp / n_find if n_find else None}


def agg(values, latex: str = "") -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "--"
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.1f}"
    lo, hi = min(values), max(values)
    if latex == "rng":
        return f"\\rng{{{mean:.1f}}}{{{lo:.1f}}}{{{hi:.1f}}}"
    return f"{mean:.1f} [{lo:.1f},{hi:.1f}]"


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    mode = sys.argv[2] if len(sys.argv) > 2 else "main"
    gold, exposed = exposure(split)
    chosen = select(split, mode)
    groups = defaultdict(list)
    provenance = []
    for (size, sup, seed), (run, path, hw) in sorted(chosen.items()):
        groups[size, sup].append(metrics(path, gold, exposed))
        provenance.append(f"{size} {sup} {seed}: {run.name} ({hw})")
    sizes = sorted({s for s, _ in groups}, key=lambda s: int(s[:-1]))
    lines = [f"# Recognition, {split} split, data {DATA}, hardware selection `{mode}` (generated by paper_recognition.py)", "",
             "| Size | Supervision | runs | Findings | Easy F1 | Medium F1 | Hard F1 | All F1 | Loc F1 | Support | Unexp. | None exp. |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    cols = ("findings", "easy", "medium", "hard", "all", "loc", "support", "unexp", "none_exp")
    for size in sizes:
        for sup in SUPS:
            ms = groups.get((size, sup))
            if ms:
                lines.append(f"| {size} | {sup} | {len(ms)} | " + " | ".join(agg([m[c] for m in ms]) for c in cols) + " |")
    lines += ["", "LaTeX rows for tables/recognition.tex (Easy, Medium, Hard, All, Loc, Evid. support):", "```"]
    for size in sizes:
        for sup in ("ordinary", "evidence"):
            ms = groups.get((size, sup))
            if ms:
                lines.append(f"{size} {sup.capitalize()} & " + " & ".join(agg([m[c] for m in ms]) for c in ("easy", "medium", "hard", "all", "loc", "support")) + r" \\")
    lines += ["```", "", "LaTeX rows for tables/recognition_sizes.tex (Findings, Easy, Medium, Hard, Loc, Support, Unexp., None exp.):", "```"]
    for size in sizes:
        for sup in SUPS:
            ms = groups.get((size, sup))
            if ms:
                lines.append(f"{size} {sup} & " + " & ".join(agg([m[c] for m in ms], "rng") for c in ("findings", "easy", "medium", "hard", "loc", "support", "unexp", "none_exp")) + r" \\")
    lines += ["```", "", "Runs used:", *[f"- {p}" for p in provenance]]
    path = OUT / f"RESULTS-recognition-paper-{split}-{mode}{'-v11' if DATA == 'v11' else ''}.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
