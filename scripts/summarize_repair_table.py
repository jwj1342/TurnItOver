"""Fill the repair table from runner outputs: success, resolved per defect type, localization, cost to success."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from turnitover.repair.loop import usage_tokens

TYPES = ("kinematics.joint_axis", "kinematics.direction_flip", "kinematics.joint_anchor", "structure.part_offset", "structure.part_rotation")


def final_audit(folder: Path) -> dict | None:
    rounds = sorted(folder.glob("round-*/audit.private.json"))
    accepted = [r for r in rounds if (r.parent / "accepted.ts").exists()]
    path = accepted[-1] if accepted else folder / "initial.audit.private.json"
    return json.loads(path.read_text()) if path.exists() else None


def resolved(audit: dict, parts: list[str]) -> bool | None:
    invariants = [i for c in audit.get("checks", []) for i in c.get("invariants", [])]
    mine = [i for i in invariants if set(i.get("parts", ())) & set(parts)] if parts else []
    if not mine:
        return bool(audit.get("passed"))
    return all(i["passed"] for i in mine)


def episode_metrics(folder: Path, labels: list[dict]) -> dict:
    result = json.loads((folder / "result.json").read_text())
    rounds = sorted(folder.glob("round-*"))
    first = result.get("first_success_round")
    tokens_to_success = cost = 0.0
    for r in rounds:
        model = r / "model.json"
        if not model.exists():
            continue
        usage = json.loads(model.read_text()).get("usage", {})
        index = int(r.name.split("-")[1])
        if first is not None and index < first:      # first_success_round counts generator calls; round k is call k+1
            tokens_to_success += usage_tokens(usage)
        try:
            cost += float(usage.get("cost") or 0.0)
        except (TypeError, ValueError):
            pass
    audit = final_audit(folder)
    per_label = [(l["defect_id"], resolved(audit, l.get("parts", [])) if audit else None) for l in labels]
    loc_total = loc_hit = 0
    for v in folder.glob("round-*/verdict.json"):
        for f in json.loads(v.read_text()).get("findings", []):
            loc_total += 1
            loc_hit += any(f["defect_id"] == l["defect_id"] and set(f.get("parts", ())) & set(l.get("parts", ())) for l in labels)
    return {"final_passed": bool(result.get("final_passed")), "first_success_round": first, "model_calls": result.get("model_calls"),
            "tokens_to_success": tokens_to_success if first is not None else None, "total_tokens": result.get("total_tokens"),
            "reported_cost_usd": cost, "non_improving_rate": result.get("non_improving_rate"),
            "exact_state_revisit_rate": result.get("exact_state_revisit_rate"), "resolved": per_label,
            "localization": (loc_hit, loc_total)}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(v, scale=1.0):
    return "--" if v is None else f"{v * scale:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, nargs="+", required=True, help="runner output directories, e.g. one per judge job")
    parser.add_argument("--cases", type=Path, required=True, help="case directory or pack root")
    parser.add_argument("--labels", type=Path, help="annotations.private.jsonl (pack input)")
    parser.add_argument("--case-ids-file", type=Path, help="score only these cases (one id per line), e.g. the eligible set")
    parser.add_argument("--summary", type=Path, help="where to write summary-table.json (default: first --out)")
    args = parser.parse_args()
    keep = None
    if args.case_ids_file:
        keep = {line.strip() for line in args.case_ids_file.read_text().splitlines() if line.strip() and not line.startswith("#")}
    labels = {}
    if args.labels:
        for line in args.labels.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["id"]] = row["labels"]
    for folder in args.cases.iterdir():
        meta = folder / "case.private.json"
        if meta.exists():
            annotation = json.loads(meta.read_text())["annotation"]
            labels[annotation["id"]] = annotation["labels"]
    rows = []
    table = {}
    judge_dirs = sorted((gen_dir, judge_dir) for out in args.out for gen_dir in out.iterdir() if gen_dir.is_dir() and gen_dir.name != "cache"
                        for judge_dir in gen_dir.iterdir() if judge_dir.is_dir())
    for gen_dir, judge_dir in judge_dirs:
        episodes = []
        for case_dir in sorted(p for p in judge_dir.iterdir() if p.is_dir() and (p / "result.json").exists()):
            if keep is not None and case_dir.name not in keep:
                continue
            if json.loads((case_dir / "result.json").read_text()).get("status") != "complete":
                continue
            episodes.append((case_dir.name, episode_metrics(case_dir, labels.get(case_dir.name, []))))
        if not episodes:
            continue
        by_type = defaultdict(list)
        loc_hit = loc_total = 0
        for _, e in episodes:
            for defect, ok in e["resolved"]:
                by_type[defect].append(ok)
            loc_hit += e["localization"][0]; loc_total += e["localization"][1]
        succ = [e for _, e in episodes if e["final_passed"]]
        summary = {"n": len(episodes), "success": len(succ) / len(episodes),
                   "resolved": {t: mean([float(x) for x in by_type[t] if x is not None]) for t in TYPES if by_type[t]},
                   "localization": loc_hit / loc_total if loc_total else None,
                   "iterations_to_success": mean([e["first_success_round"] for e in succ]),
                   "tokens_to_success_k": mean([e["tokens_to_success"] / 1000 for e in succ]),
                   "total_tokens_k": sum(e["total_tokens"] or 0 for _, e in episodes) / 1000,
                   "reported_cost_usd": sum(e["reported_cost_usd"] for _, e in episodes),
                   "non_improving_rate": mean([e["non_improving_rate"] for _, e in episodes]),
                   "exact_state_revisit_rate": mean([e["exact_state_revisit_rate"] for _, e in episodes])}
        key = f"{gen_dir.name}/{judge_dir.name}"
        if key in table:
            raise SystemExit(f"{key} appears in more than one --out directory")
        table[key] = summary
        rows.append((gen_dir.name, judge_dir.name, summary))
    (args.summary or args.out[0] / "summary-table.json").write_text(json.dumps(table, indent=2))
    print("| Generator | Judge | n | Success | " + " | ".join(t.split(".")[1] for t in TYPES) + " | Loc. | Iter. | Tokens (k) | Non-impr. | Revisit | Cost $ |")
    print("|---|---|--:|" + "--:|" * 11)
    for gen, judge, s in rows:
        print(f"| {gen} | {judge} | {s['n']} | {fmt(s['success'], 100)} | " + " | ".join(fmt(s['resolved'].get(t), 100) for t in TYPES)
              + f" | {fmt(s['localization'], 100)} | {fmt(s['iterations_to_success'])} | {fmt(s['tokens_to_success_k'])} | "
              + f"{fmt(s['non_improving_rate'], 100)} | {fmt(s['exact_state_revisit_rate'], 100)} | {s['reported_cost_usd']:.2f} |")


if __name__ == "__main__":
    main()
