"""LaTeX rows for tables/repair.tex, repair_breakdown.tex and repair_cost.tex from summarize_repair_table.py's JSON.

One generator per call (the generator column of repair.tex holds that generator's Success; the other columns are
that generator's numbers, not a two-generator average). Several summaries (one per run: generator repetitions for the
fixed judges, training seeds for the ours_* judges) are averaged per judge over the runs that contain it; Success is
printed as the mean with the [min, max] over runs in the comment lines. Judges without runs are printed as N/A rows. Resolved needs
motion / visible at rest pools labels by the taxonomy's observation requirement (summary fields resolved_motion and
resolved_rest). Inv. in repair_cost.tex is the physical-invariant pass rate of the final program (summary field
physical_invariant_rate: share of geometry.alignment invariants the final program satisfies, mean over complete episodes).
Usage: python scripts/exp900/analysis/paper_repair.py <generator id> <summary.json> [<summary.json> ...]
"""
from __future__ import annotations

import json
import os
import sys

ROWS = [("none", "No verifier (fixed iterations)"), ("pixel", "pixel difference"),
        ("zs4b_screenshot", "zero-shot Qwen3-VL-4B, at rest"), ("zs4b_sweep", "zero-shot Qwen3-VL-4B, fixed scan"),
        ("zs8b_screenshot", "zero-shot Qwen3-VL-8B, at rest"), ("zs8b_sweep", "zero-shot Qwen3-VL-8B, fixed scan"),
        ("claude_sweep", "zero-shot Claude Fable 5.1, fixed scan"), ("zsapi_sweep", "zero-shot Qwen3.8-27B, fixed scan"),
        ("ours_2b_ordinary", r"\paper{} 2B, Ordinary"), ("ours_2b_evidence", r"\paper{} 2B, Evidence"),
        ("ours_8b_evidence", r"\paper{} 8B, Evidence"), ("gold", "Ground-truth diagnosis")]
TYPES = ("kinematics.joint_axis", "kinematics.direction_flip", "kinematics.joint_anchor", "structure.part_offset",
         "structure.part_rotation")


def pct(v) -> str:
    return "--" if v is None else f"{100 * v:.1f}"


def num(v) -> str:
    return "--" if v is None else f"{v:.1f}"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def combine(runs: list[dict], key: str):
    rows = [r[key] for r in runs if key in r]
    if not rows:
        return None
    out = {f: mean([r[f] for r in rows]) for f in ("success", "resolved_motion", "resolved_rest", "localization",
                                                   "iterations_to_success", "tokens_to_success_k", "non_improving_rate",
                                                   "exact_state_revisit_rate", "physical_invariant_rate")}
    out["resolved"] = {t: mean([r["resolved"].get(t) for r in rows]) for t in TYPES}
    out["runs"] = rows
    return out


def main() -> None:
    gen, runs = sys.argv[1], [json.loads(open(path).read()) for path in sys.argv[2:]]
    table = {f"{gen}/{judge}": combine(runs, f"{gen}/{judge}") for judge, _ in ROWS}
    table = {k: v for k, v in table.items() if v is not None}
    out = [f"% generator {gen}; Success with Wilson 95% CI in the markdown summary; n per judge = eligible cases",
           "% repair.tex: Judge & Success & Resolved needs motion & Resolved visible at rest & Iter. & Tokens (k)"]
    for judge, label in ROWS:
        s = table.get(f"{gen}/{judge}")
        if s is None:
            continue
        iters = "--" if judge == "none" else num(s["iterations_to_success"])
        out.append(f"{label} & {pct(s['success'])} & {pct(s['resolved_motion'])} & {pct(s['resolved_rest'])} & {iters} & "
                   f"{num(s['tokens_to_success_k'])} \\\\")
    boot_path = os.environ.get("TIO_BOOTSTRAP_JSON")
    if boot_path:
        boot = json.loads(open(boot_path).read())
        out.append("% table1: Verifier & Success [95% CI] & vs none [95% CI] & Visible after motion & Visible at rest & Iter. & Tokens (k)")
        for judge, label in ROWS:
            s, b = table.get(f"{gen}/{judge}"), boot.get(judge)
            if s is None or b is None:
                continue
            lo, hi = b["ci"]
            d = "--" if "diff_none" not in b else (f"{b['diff_none']:+.1f}" + r"\,{\tiny[%+.1f, %+.1f]}" % tuple(b["diff_none_ci"]))
            iters = "--" if judge == "none" else num(s["iterations_to_success"])
            out.append(f"{label} & {b['success']:.1f}" + r"\,{\tiny[%.1f, %.1f]}" % (lo, hi) + f" & {d} & {pct(s['resolved_motion'])} & "
                       f"{pct(s['resolved_rest'])} & {iters} & {num(s['tokens_to_success_k'])} \\\\")
    out.append("% repair_breakdown.tex: Judge & Generator & Success & Axis & Flip & Anchor & Offset & Rotation & Loc. & Iter. & Tokens (k)")
    for judge, label in ROWS:
        s = table.get(f"{gen}/{judge}")
        if s is None:
            continue
        loc = "--" if judge in ("none", "pixel") else pct(s["localization"])
        out.append(f"{label} & {gen} & {pct(s['success'])} & " + " & ".join(pct(s["resolved"].get(t)) for t in TYPES)
                   + f" & {loc} & {num(s['iterations_to_success'])} & {num(s['tokens_to_success_k'])} \\\\")
    out.append("% repair_cost.tex: Judge & Non-impr. & Osc. & Inv. (share of geometry invariants the final program passes)")
    for judge, label in ROWS:
        s = table.get(f"{gen}/{judge}")
        if s is None:
            continue
        out.append(f"{label} & {pct(s['non_improving_rate'])} & {pct(s['exact_state_revisit_rate'])} & {pct(s['physical_invariant_rate'])} \\\\")
    out.append("% per run: n, complete, missing, error, success with Wilson 95% CI, invalid-patch rate, resolved counts")
    for judge, _ in ROWS:
        c = table.get(f"{gen}/{judge}")
        if c is None:
            continue
        rates = [r["success"] for r in c["runs"]]
        out.append(f"% {judge}: {len(rates)} runs, success mean {pct(c['success'])} [{pct(min(rates))}, {pct(max(rates))}]")
        for s in c["runs"]:
            lo, hi = s["success_ci95"]
            out.append(f"%   n={s['n']} complete={s['n_complete']} missing={s['n_missing']} error={s['n_error']} "
                       f"success={pct(s['success'])} [{pct(lo)}, {pct(hi)}] invalid_patch={pct(s['invalid_patch_rate'])} "
                       f"resolved_counts={s['resolved_counts']}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
