"""LaTeX rows for tables/repair.tex, repair_breakdown.tex and repair_cost.tex from summarize_repair_table.py's JSON.

One generator per summary (the generator column of repair.tex holds that generator's Success; the other columns are
that generator's numbers, not a two-generator average). Judges without runs are printed as N/A rows. Resolved needs
motion / visible at rest pools labels by the taxonomy's observation requirement (summary fields resolved_motion and
resolved_rest). Inv. in repair_cost.tex is the physical-invariant pass rate of the final program (summary field
physical_invariant_rate: share of geometry.alignment invariants the final program satisfies, mean over complete episodes).
Usage: python scripts/exp900/analysis/paper_repair.py <summary.json> <generator id>
"""
from __future__ import annotations

import json
import sys

ROWS = [("none", "none (fixed rounds)"), ("pixel", "pixel difference"),
        ("zs8b_screenshot", "zero-shot Qwen3-VL-8B, single screenshot"), ("zs8b_sweep", "zero-shot Qwen3-VL-8B, fixed sweep"),
        ("claude_sweep", "zero-shot Claude Fable 5.1, fixed sweep"),
        ("ours_2b_ordinary", r"\paper{} 2B, Ordinary"), ("ours_2b_evidence", r"\paper{} 2B, Evidence"),
        ("ours_8b_evidence", r"\paper{} 8B, Evidence"), ("gold", "gold findings (upper bound)")]
TYPES = ("kinematics.joint_axis", "kinematics.direction_flip", "kinematics.joint_anchor", "structure.part_offset",
         "structure.part_rotation")


def pct(v) -> str:
    return "--" if v is None else f"{100 * v:.1f}"


def num(v) -> str:
    return "--" if v is None else f"{v:.1f}"


def main() -> None:
    table, gen = json.loads(open(sys.argv[1]).read()), sys.argv[2]
    out = [f"% generator {gen}; Success with Wilson 95% CI in the markdown summary; n per judge = eligible cases",
           "% repair.tex: Judge & Success & Resolved needs motion & Resolved visible at rest & Iter. & Tokens (k)"]
    for judge, label in ROWS:
        s = table.get(f"{gen}/{judge}")
        if s is None:
            out.append(f"{label} & N/A & N/A & N/A & N/A & N/A \\\\")
            continue
        iters = "--" if judge == "none" else num(s["iterations_to_success"])
        out.append(f"{label} & {pct(s['success'])} & {pct(s['resolved_motion'])} & {pct(s['resolved_rest'])} & {iters} & "
                   f"{num(s['tokens_to_success_k'])} \\\\")
    out.append("% repair_breakdown.tex: Judge & Generator & Success & Axis & Flip & Anchor & Offset & Rotation & Loc. & Iter. & Tokens (k)")
    for judge, label in ROWS:
        s = table.get(f"{gen}/{judge}")
        if s is None:
            out.append(f"{label} & {gen} & " + " & ".join(["N/A"] * 9) + " \\\\")
            continue
        loc = "--" if judge in ("none", "pixel") else pct(s["localization"])
        out.append(f"{label} & {gen} & {pct(s['success'])} & " + " & ".join(pct(s["resolved"].get(t)) for t in TYPES)
                   + f" & {loc} & {num(s['iterations_to_success'])} & {num(s['tokens_to_success_k'])} \\\\")
    out.append("% repair_cost.tex: Judge & Non-impr. & Osc. & Inv. (share of geometry invariants the final program passes)")
    for judge, label in ROWS:
        s = table.get(f"{gen}/{judge}")
        if s is None:
            out.append(f"{label} & N/A & N/A & N/A \\\\")
            continue
        out.append(f"{label} & {pct(s['non_improving_rate'])} & {pct(s['exact_state_revisit_rate'])} & {pct(s['physical_invariant_rate'])} \\\\")
    out.append("% supporting counts per judge: n, complete, missing, error, success CI95, invalid-patch rate, resolved counts")
    for judge, _ in ROWS:
        s = table.get(f"{gen}/{judge}")
        if s:
            lo, hi = s["success_ci95"]
            out.append(f"% {judge}: n={s['n']} complete={s['n_complete']} missing={s['n_missing']} error={s['n_error']} "
                       f"success={pct(s['success'])} [{pct(lo)}, {pct(hi)}] invalid_patch={pct(s['invalid_patch_rate'])} "
                       f"resolved_counts={s['resolved_counts']}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
