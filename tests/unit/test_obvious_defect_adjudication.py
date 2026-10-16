import json

from scripts.adjudicate_obvious_defect_review import DEFAULT_BUNDLE, DEFAULT_REVIEW, adjudicate


def test_human_review_adjudication_is_complete_and_bounded():
    results, summary = adjudicate(DEFAULT_BUNDLE, DEFAULT_REVIEW)
    assert len(results) == summary["cases"] == 9
    assert summary["submission_rows"] == 10
    assert summary["duplicate_ids"] == {"odr-902e87011e30": 2}
    assert summary["conflicting_duplicate_ids"] == []
    assert summary["counts"] == {"needs_adjudication": 3, "review_approved_candidate": 6}
    assert summary["by_gold_kind"] == {
        "clean": {"needs_adjudication": 1, "review_approved_candidate": 2},
        "joint_axis": {"review_approved_candidate": 3},
        "part_offset": {"needs_adjudication": 2, "review_approved_candidate": 1},
    }
    assert summary["boundaries"] == {
        "sft_exported": False, "sft_trained": False, "test_modified": False, "model_calls": 0}
    assert all(row["checker_consistent"] for row in results)
    assert all(row["field_matches"] == {"verdict": True, "defect_id": True, "joint_id": True, "parts": True}
               for row in results if row["human_status"] == "human_review_complete")
    assert {row["case_id"] for row in results if row["candidate_status"] == "needs_adjudication"} == {
        "odr-0475067092fd", "odr-485c93213d2f", "odr-902e87011e30"}


def test_saved_adjudication_matches_recomputed_result():
    expected, summary = adjudicate(DEFAULT_BUNDLE, DEFAULT_REVIEW)
    saved = [json.loads(line) for line in
             (DEFAULT_REVIEW / "private/adjudication.jsonl").read_text().splitlines()]
    assert saved == expected
    assert json.loads((DEFAULT_REVIEW / "summary.json").read_text()) == summary
