"""Join saved teacher Episodes, their independent correctness audit, and the perceptibility sweep.

Offline only: reads ``episode.json`` / ``audit.correctness-v1.json`` beside each saved real
Episode and the ``summary.json`` written by ``sweep_corruption_perceptibility``. Each defect
Episode receives a rule-based provisional attribution; the rule uses the frozen pixel
perceptibility tier, so it is a proxy that the report must confirm or overturn by looking
at the images:

  1 corruption_imperceptible   no state in the allowed action space passes the perceptible rule
  2 coverage_insufficient      perceptible states exist but the trajectory credited none of them
  3 misjudged_with_evidence    credited evidence was acquired, yet the verdict is pass/uncertain/wrong type
  4 localization_or_citation   right defect type, but wrong joint/parts or cited steps without credited evidence
  correct_fail / no_legal_verdict  are reported as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = "audit.correctness-v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attribute(audit: dict, case_has_perceptible: bool) -> str:
    outcome = audit["diagnosis_outcome"]
    if outcome == "no_legal_verdict":
        return "no_legal_verdict"
    if outcome == "correct_fail":
        return "correct_fail"
    if not case_has_perceptible:
        return "1_corruption_imperceptible"
    if not audit["objective_evidence_covered"]:
        return "2_coverage_insufficient"
    if audit["status"] in ("pass", "uncertain"):
        return "3_misjudged_with_evidence"
    if not any(loc["defect_type_correct"] for loc in audit["localization"]):
        return "3_misjudged_with_evidence"
    return "4_localization_or_citation"


def episode_row(path: Path, sweep_case: dict) -> dict:
    episode = json.loads((path / "episode.json").read_text())
    audit_file = path / AUDIT
    audit = json.loads(audit_file.read_text())["independent_evidence_audit"]
    identity = episode.get("data_identity", {})
    steps = []
    for row in audit["action_audit"]:
        step = {"step": row["step"], "source": row.get("decision_source"),
                "state": row.get("state"), "bound": row.get("bound"),
                "reference_comparable": row.get("reference_comparable"),
                "exposures": row.get("exposures"), "perceptible_exposures": row.get("perceptible_exposures"),
                "witness_keys": row.get("witness_keys"), "reason": row.get("reason")}
        record = next(r for r in episode["trajectory"] if r["step"] == row["step"])
        step["decision"] = record["decision"]
        if record.get("image_ref"):
            step["image_ref"] = record["image_ref"]
            step["image_sha256"] = row.get("live_sha256")
        steps.append(step)
    best = max((r["core_changed_fraction"] for r in sweep_case["current"]), default=0.0)
    perceptible_states = [r for r in sweep_case["current"] if r["perceptible_rule"]]
    visited = {(s["state"]["view"], json.dumps(s["state"]["joints"], sort_keys=True))
               for s in steps if s.get("state")}
    visited_core = [r["core_changed_fraction"] for r in sweep_case["current"]
                    if (r["view"], json.dumps(r["state"], sort_keys=True)) in visited]
    verdict = episode.get("verdict") or {}
    return {
        "dir": path.relative_to(ROOT).as_posix(),
        "round": int(re.search(r"round(\d)", path.name).group(1)),
        "mode": identity.get("experiment_mode"), "reference_protocol": identity.get("reference_protocol"),
        "common_input": identity.get("common_input"), "prompt_addenda": identity.get("prompt_addenda"),
        "budget": episode.get("budget"), "spent": episode.get("spent"), "termination": episode.get("termination"),
        "status": audit["status"], "teacher_confidence": verdict.get("confidence"),
        "teacher_summary": verdict.get("summary"),
        "teacher_findings": verdict.get("findings", []),
        "localization": audit["localization"], "diagnosis_outcome": audit["diagnosis_outcome"],
        "accepted": audit["accepted"], "rejection_reasons": audit["rejection_reasons"],
        "objective_evidence_covered": audit["objective_evidence_covered"],
        "objective_exposure_imperceptible_only": audit["objective_exposure_imperceptible_only"],
        "steps": steps,
        "visited_max_core_fraction": max(visited_core, default=0.0),
        "action_space_max_core_fraction": best,
        "action_space_perceptible_states": len(perceptible_states),
        "attribution": attribute(audit, bool(perceptible_states)),
        "episode_sha256": _sha(path / "episode.json"), "audit_sha256": _sha(audit_file),
    }


def _pair(row: dict) -> dict:
    return {"view": row["view"], "state": row["state_label"],
            **{k: row[k] for k in ("changed_fraction", "core_changed_fraction", "largest_core_region_px",
                                   "perceptible_rule", "reference_motion_changed_fraction")},
            "px_max": row["displacement"]["px_max"], "world_max": row["displacement"]["world_max"]}


def case_summary(case: dict, defect: list[dict], controls: list[dict]) -> dict:
    """Action-space coverage of the current corruption, the strength gradient, and teacher outcomes."""
    current = case["current"]
    by_view = {}
    for row in current:
        by_view.setdefault(row["view"], [0, 0])
        by_view[row["view"]][0] += row["perceptible_rule"]
        by_view[row["view"]][1] += 1
    gradient = []
    for step in case["gradient"]:
        best = max(step["rows"], key=lambda r: r["core_changed_fraction"])
        gradient.append({"angle_deg": step["angle_deg"],
                         "perceptible_pairs": sum(r["perceptible_rule"] for r in step["rows"]),
                         "pairs": len(step["rows"]), "best": _pair(best)})
    episodes = [e for e in defect if e["id"] == case["id"]]
    count = lambda key, rows: {v: sum(r[key] == v for r in rows) for v in sorted({r[key] for r in rows})}
    return {"id": case["id"], "asset_id": case["asset_id"], "category": case["category"],
            "joint": case["joint"], "joint_type": case["joint_type"], "joint_limits": case["joint_limits"],
            "actual_angle_deg": case["actual_angle_deg"], "framing_radius": case["framing"]["radius"],
            "frozen_witnesses_verified": case["frozen_witnesses_verified"],
            "action_space_pairs": len(current),
            "action_space_perceptible_pairs": sum(r["perceptible_rule"] for r in current),
            "perceptible_by_view": {v: f"{a}/{b}" for v, (a, b) in by_view.items()},
            "best_pairs": [_pair(r) for r in sorted(current, key=lambda r: -r["core_changed_fraction"])[:3]],
            "gradient_views": case["gradient_views"], "gradient": gradient,
            "teacher_attribution": count("attribution", episodes),
            "teacher_outcomes": count("diagnosis_outcome", episodes),
            "clean_control_outcomes": count("diagnosis_outcome", [c for c in controls
                                                                  if c["asset_id"] == case["asset_id"]])}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True, help="summary.json from the sweep")
    parser.add_argument("--episodes-root", type=Path, default=ROOT / "output/interactive-teacher")
    parser.add_argument("--clean-id", action="append", default=[], help="ASSET=clean episode id")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.sweep, args.episodes_root = args.sweep.resolve(), args.episodes_root.resolve()
    sweep = {case["id"]: case for case in json.loads(args.sweep.read_text())}
    clean = dict(item.split("=", 1)[::-1] for item in args.clean_id)
    defect, controls = [], []
    for path in sorted(p.parent for p in args.episodes_root.glob("kimi-round*/episode.json")):
        episode_id = json.loads((path / "episode.json").read_text()).get("episode_id")
        if episode_id in sweep:
            defect.append({"id": episode_id, "asset_id": sweep[episode_id]["asset_id"],
                           **episode_row(path, sweep[episode_id])})
        elif episode_id in clean:
            audit = json.loads((path / AUDIT).read_text())["independent_evidence_audit"]
            controls.append({"id": episode_id, "asset_id": clean[episode_id],
                             "dir": path.relative_to(ROOT).as_posix(), "status": audit["status"],
                             "diagnosis_outcome": audit["diagnosis_outcome"], "accepted": audit["accepted"]})
    result = {"sweep": {"path": args.sweep.relative_to(ROOT).as_posix(), "sha256": _sha(args.sweep)},
              "cases": [case_summary(case, defect, controls) for case in sweep.values()],
              "defect_episodes": defect, "clean_controls": controls}
    args.out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return result


if __name__ == "__main__":
    main()
