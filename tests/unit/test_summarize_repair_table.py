"""Repair table summary: fixed case-list denominator, Wilson CI, resolved from the last applied patch, applied-only stability."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("summarize_repair_table", Path(__file__).parents[2] / "scripts/summarize_repair_table.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def audit(passed: bool, part: str) -> dict:
    return {"passed": passed, "checks": [{"invariants": [{"parts": [part], "passed": passed}]}]}


def episode(folder: Path, result: dict, audits: dict[str, dict]) -> None:
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps(result))
    for name, value in audits.items():
        (folder / name).parent.mkdir(exist_ok=True)
        (folder / name).write_text(json.dumps(value))


def round_record(index, outcome, before, after, after_sha):
    return {"round": index, "outcome": outcome, "before_score": before, "after_score": after, "after_sha256": after_sha}


def test_summary_counts_every_listed_case_and_uses_last_applied_audit(tmp_path, monkeypatch):
    judge = tmp_path / "out" / "gen" / "gold"
    # a: applied fix, then an unusable proposal whose audit fails; the accepted program is the fixed one
    episode(judge / "a", {"status": "complete", "final_passed": True, "first_success_round": 1, "initial_sha256": "s0",
                          "delta": 0.0, "total_tokens": 10,
                          "rounds": [round_record(0, "applied", 0.0, 1.0, "s1"), round_record(1, "unusable_proposal", 1.0, 1.0, "s1")]},
            {"initial.audit.private.json": audit(False, "p"), "round-000/audit.private.json": audit(True, "p"),
             "round-001/audit.private.json": audit(False, "p")})
    episode(judge / "b", {"status": "generator_error", "error": "HTTP 503"}, {})
    episode(judge / "d", {"status": "complete", "final_passed": False, "first_success_round": None, "initial_sha256": "s0",
                          "delta": 0.0, "total_tokens": 5,
                          "rounds": [round_record(0, "invalid_patch", 0.0, 0.0, "s0"), round_record(1, "stop", 0.0, 0.0, "s0")]},
            {"initial.audit.private.json": audit(False, "q")})
    episode(judge / "e", {"status": "incomplete"}, {})
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({"id": i, "labels": [{"defect_id": d, "parts": [p]}]}) + "\n" for i, d, p in
                              [("a", "kinematics.joint_axis", "p"), ("b", "kinematics.joint_axis", "p"),
                               ("c", "kinematics.joint_axis", "p"), ("d", "structure.part_offset", "q"),
                               ("e", "structure.part_offset", "q")]))
    ids = tmp_path / "ids.txt"
    ids.write_text("# eligible\na\nb\nc\nd\ne\n")
    (tmp_path / "cases").mkdir()
    monkeypatch.setattr(sys, "argv", ["summarize", "--out", str(tmp_path / "out"), "--cases", str(tmp_path / "cases"),
                                      "--labels", str(labels), "--case-ids-file", str(ids), "--summary", str(tmp_path / "s.json")])
    summary.main()
    row = json.loads((tmp_path / "s.json").read_text())["gen/gold"]
    assert (row["n"], row["n_complete"], row["n_missing"], row["n_error"]) == (5, 2, 2, 1)
    assert row["success"] == pytest.approx(0.2)
    assert row["success_ci95"] == pytest.approx(list(summary.wilson(1, 5)))
    assert row["success_ci95"] == pytest.approx([0.0362, 0.6245], abs=1e-4)
    assert row["resolved"] == {"kinematics.joint_axis": pytest.approx(1 / 3), "structure.part_offset": 0.0}
    assert row["invalid_patch_rate"] == pytest.approx(1 / 3) and row["patch_rounds"] == 3
    assert (row["applied_rounds"], row["non_improving_rate"], row["exact_state_revisit_rate"]) == (1, 0.0, 0.0)
