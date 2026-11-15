"""Fill the repair table from runner outputs: success with Wilson 95% CI, resolved per defect type, localization,
invalid patches, cost to success and stability over applied patches.

Every judge is scored on the same case list (--case-ids-file). Missing, incomplete and generator_error episodes count as
failures for Success and leave their labels unresolved; they are reported per judge as n_missing (missing or incomplete)
and n_error (generator_error). The remaining columns pool the complete episodes: Loc. over findings, Inv. over generator
rounds that returned a patch, Non-impr. and Revisit over rounds whose patch was applied.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from turnitover.repair.loop import stability, usage_tokens

TYPES = ("kinematics.joint_axis", "kinematics.direction_flip", "kinematics.joint_anchor", "structure.part_offset", "structure.part_rotation")
Z95 = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    centre = (p + Z95 ** 2 / (2 * n)) / (1 + Z95 ** 2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / (1 + Z95 ** 2 / n)
    return centre - half, centre + half


def final_audit(folder: Path, result: dict) -> dict:
    """Audit of the last round whose patch was applied, else of the initial program."""
    applied = [r["round"] for r in result["rounds"] if r["outcome"] == "applied"]
    path = folder / f"round-{applied[-1]:03d}" / "audit.private.json" if applied else folder / "initial.audit.private.json"
    return json.loads(path.read_text())


def resolved(audit: dict, parts: list[str]) -> bool:
    invariants = [i for c in audit.get("checks", []) for i in c.get("invariants", [])]
    mine = [i for i in invariants if set(i.get("parts", ())) & set(parts)] if parts else []
    if not mine:
        return bool(audit.get("passed"))
    return all(i["passed"] for i in mine)


def episode_metrics(folder: Path, result: dict, labels: list[dict]) -> dict:
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
    audit = final_audit(folder, result)
    loc_total = loc_hit = 0
    for v in folder.glob("round-*/verdict.json"):
        for f in json.loads(v.read_text()).get("findings", []):
            loc_total += 1
            loc_hit += any(f["defect_id"] == l["defect_id"] and set(f.get("parts", ())) & set(l.get("parts", ())) for l in labels)
    outcomes = [r["outcome"] for r in result["rounds"]]
    return {"final_passed": bool(result["final_passed"]), "first_success_round": first,
            "tokens_to_success": tokens_to_success if first is not None else None, "total_tokens": result.get("total_tokens"),
            "reported_cost_usd": cost, "resolved": [(l["defect_id"], resolved(audit, l.get("parts", []))) for l in labels],
            "localization": (loc_hit, loc_total),
            "invalid": (outcomes.count("invalid_patch"), len(outcomes) - outcomes.count("stop")),
            "stability": stability(result["rounds"], result["initial_sha256"], result["delta"])}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def ratio(hit: int, total: int):
    return hit / total if total else None


def fmt(v, scale=1.0):
    return "--" if v is None else f"{v * scale:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, nargs="+", required=True, help="runner output directories, e.g. one per judge job")
    parser.add_argument("--cases", type=Path, required=True, help="case directory or pack root")
    parser.add_argument("--labels", type=Path, help="annotations.private.jsonl (pack input)")
    parser.add_argument("--case-ids-file", type=Path, required=True,
                        help="the denominator: score exactly these cases (one id per line, # comments), e.g. the eligible set")
    parser.add_argument("--summary", type=Path, help="where to write summary-table.json (default: first --out)")
    args = parser.parse_args()
    case_ids = [line.strip() for line in args.case_ids_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
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
        n_missing = n_error = 0
        by_type = defaultdict(list)
        for case in case_ids:
            path = judge_dir / case / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
            if result["status"] == "complete":
                episode = episode_metrics(judge_dir / case, result, labels.get(case, []))
                episodes.append(episode)
                for defect, ok in episode["resolved"]:
                    by_type[defect].append(ok)
                continue
            if result["status"] == "generator_error":
                n_error += 1
            else:
                n_missing += 1
            for label in labels.get(case, []):
                by_type[label["defect_id"]].append(False)
        succ = [e for e in episodes if e["final_passed"]]
        applied = sum(e["stability"]["applied"] for e in episodes)
        patches = sum(e["invalid"][1] for e in episodes)
        summary = {"n": len(case_ids), "n_complete": len(episodes), "n_missing": n_missing, "n_error": n_error,
                   "success": len(succ) / len(case_ids), "success_ci95": wilson(len(succ), len(case_ids)),
                   "resolved": {t: mean([float(x) for x in by_type[t]]) for t in TYPES if by_type[t]},
                   "localization": ratio(sum(e["localization"][0] for e in episodes), sum(e["localization"][1] for e in episodes)),
                   "invalid_patch_rate": ratio(sum(e["invalid"][0] for e in episodes), patches), "patch_rounds": patches,
                   "iterations_to_success": mean([e["first_success_round"] for e in succ]),
                   "tokens_to_success_k": mean([e["tokens_to_success"] / 1000 for e in succ]),
                   "total_tokens_k": sum(e["total_tokens"] or 0 for e in episodes) / 1000,
                   "reported_cost_usd": sum(e["reported_cost_usd"] for e in episodes),
                   "non_improving_rate": ratio(sum(e["stability"]["non_improving"] for e in episodes), applied),
                   "exact_state_revisit_rate": ratio(sum(e["stability"]["revisits"] for e in episodes), applied),
                   "applied_rounds": applied}
        key = f"{gen_dir.name}/{judge_dir.name}"
        if key in table:
            raise SystemExit(f"{key} appears in more than one --out directory")
        table[key] = summary
        rows.append((gen_dir.name, judge_dir.name, summary))
    (args.summary or args.out[0] / "summary-table.json").write_text(json.dumps(table, indent=2))
    print("| Generator | Judge | n | Missing | Error | Success | 95% CI | " + " | ".join(t.split(".")[1] for t in TYPES)
          + " | Loc. | Inv. | Iter. | Tokens (k) | Non-impr. | Revisit | Cost $ |")
    print("|---|---|" + "--:|" * 17)
    for gen, judge, s in rows:
        low, high = s["success_ci95"]
        print(f"| {gen} | {judge} | {s['n']} | {s['n_missing']} | {s['n_error']} | {fmt(s['success'], 100)} | "
              f"{fmt(low, 100)}-{fmt(high, 100)} | " + " | ".join(fmt(s['resolved'].get(t), 100) for t in TYPES)
              + f" | {fmt(s['localization'], 100)} | {fmt(s['invalid_patch_rate'], 100)} | {fmt(s['iterations_to_success'])} | "
              + f"{fmt(s['tokens_to_success_k'])} | {fmt(s['non_improving_rate'], 100)} | {fmt(s['exact_state_revisit_rate'], 100)} | "
              + f"{s['reported_cost_usd']:.2f} |")


if __name__ == "__main__":
    main()
