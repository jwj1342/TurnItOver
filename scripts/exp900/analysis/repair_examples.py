"""Candidate qualitative repair episodes for the paper's example figure: within one run, cases that an evidence-trained
judge repaired while the no-feedback run of the same case did not, with what the judge reported, the code change and the
checker trace. Also lists each case's outcome under every judge of that run.
Usage: python scripts/exp900/analysis/repair_examples.py <run dir, e.g. tio-runs/repair/q38/pilot> <generator id> [max examples]
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LABELS = ROOT / "tio-runs/data900/three-tier-900-sft-v1/episode_export/annotations.private.jsonl"
ELIGIBLE = ROOT / "tio-runs/repair/eligible.txt"
JUDGES = ("none", "pixel", "zs8b_screenshot", "zs8b_sweep", "ours_2b_ordinary", "ours_2b_evidence", "ours_8b_evidence", "gold")


def outcome(folder: Path) -> str:
    path = folder / "result.json"
    if not path.exists():
        return "missing"
    result = json.loads(path.read_text())
    if result["status"] != "complete":
        return result["status"]
    return "pass" if result["final_passed"] else "fail"


def code_change(folder: Path) -> list[str]:
    before = (folder / "initial.ts").read_text().splitlines()
    after = (folder / "final.ts").read_text().splitlines()
    lines = [l[:160] for l in difflib.unified_diff(before, after, lineterm="", n=0) if not l.startswith(("---", "+++"))]
    return lines[:16] + (["..."] if len(lines) > 16 else [])


def main() -> None:
    run, gen = Path(sys.argv[1]), sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    labels = {}
    for line in LABELS.read_text().splitlines():
        row = json.loads(line)
        labels[row["id"]] = row
    cases = [l.strip() for l in ELIGIBLE.read_text().splitlines() if l.strip() and not l.startswith("#")]
    folder = lambda judge, case: run / judge / gen / judge / case
    out = [f"# Repair episode candidates ({run.name}, generator {gen})", "",
           "| Case | Tier | Defects | " + " | ".join(JUDGES) + " |", "|---|---|---|" + "---|" * len(JUDGES)]
    for case in cases:
        defects = ", ".join(l["defect_id"].split(".")[1] for l in labels[case]["labels"])
        out.append(f"| {case} | {labels[case].get('difficulty', '')} | {defects} | "
                   + " | ".join(outcome(folder(j, case)) for j in JUDGES) + " |")
    shown = 0
    for case in cases:
        if outcome(folder("none", case)) != "fail":
            continue
        for judge in ("ours_2b_evidence", "ours_8b_evidence"):
            episode = folder(judge, case)
            if outcome(episode) != "pass" or shown >= limit:
                continue
            shown += 1
            result = json.loads((episode / "result.json").read_text())
            verdict = json.loads((episode / "round-000" / "verdict.json").read_text())
            feedback = json.loads((episode / "round-000" / "feedback.json").read_text())
            out += ["", f"## {case} ({judge})", "",
                    f"- Injected: " + "; ".join(f"{l['defect_id']} on {', '.join(l.get('parts', []))}" for l in labels[case]["labels"]),
                    "- Observations given to judge and generator: " + ", ".join(f"{o['kind']} {o['view']}/{o['state']}" for o in feedback["observations"]),
                    f"- Judge verdict (round 0): {verdict['status']}; findings: "
                    + ("; ".join(f"{f['defect_id']} on {', '.join(f.get('parts') or [])} citing steps {f.get('evidence_steps')}" for f in verdict["findings"]) or "none"),
                    f"- Checker score trace: {[round(s, 3) for s in result['score_trace']]}; termination {result['termination']}; "
                    f"generator calls {result['model_calls']}",
                    f"- No-feedback run of the same case: {outcome(folder('none', case))}, score trace "
                    f"{[round(s, 3) for s in json.loads((folder('none', case) / 'result.json').read_text())['score_trace']]}",
                    "", "```diff", *code_change(episode), "```"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
