import json

from PIL import Image

from scripts.run_kimi_obvious_axis_diagnosis import (
    BOARD_X,
    BOARD_Y,
    STATES,
    TILE,
    _approved_cases,
    _crop_board,
    _expected_steps,
    _score,
)


def _episode(status, findings=()):
    return {"teacher_verdict_missing": False,
            "verdict": {"status": status, "confidence": 0.8, "findings": list(findings),
                        "summary": "diagnosis", "limitations": []}}


def test_selection_is_three_approved_axis_clean_pairs():
    cases = _approved_cases()
    assert len(cases) == 6
    assert [case["gold"]["kind"] for case in cases] == ["clean", "joint_axis"] * 3
    assert len({case["public"]["asset_id"] for case in cases}) == 3
    axes = [case for case in cases if case["gold"]["kind"] == "joint_axis"]
    controls = [case for case in cases if case["gold"]["kind"] == "clean"]
    assert all(case["human_gold_agree"] and case["human_review_gate"] == "passed" for case in axes)
    assert {case["human"]["verdict"] for case in controls} == {"pass", "uncertain"}


def test_board_crop_uses_native_tiles_and_records_hashes(tmp_path):
    case = _approved_cases()[0]
    references, candidates, records = _crop_board(case, tmp_path)
    assert len(references) == len(candidates) == 3
    assert [row["state"] for row in records[:3]] == list(STATES)
    assert all(Image.open(path).size == (TILE, TILE) for path in references + candidates)
    assert records[0]["source_crop"] == [BOARD_X[0], BOARD_Y["reference"],
                                          BOARD_X[0] + TILE, BOARD_Y["reference"] + TILE]
    assert all(len(row["sha256"]) == 64 for row in records)


def test_axis_scoring_is_strict_over_all_requested_fields():
    case = next(case for case in _approved_cases() if case["gold"]["kind"] == "joint_axis")
    steps = _expected_steps(case)
    finding = {"defect_id": case["gold"]["defect_id"], "joint_id": case["gold"]["joint_id"],
               "parts": case["gold"]["parts"], "evidence_steps": steps}
    correct = _score(case, _episode("fail", [finding]))
    assert correct["strict_axis_correct"] is True
    assert correct["error_category"] == "strict_correct"

    wrong_steps = _score(case, _episode("fail", [{**finding, "evidence_steps": [steps[-1]]}]))
    assert wrong_steps["strict_axis_correct"] is False
    assert wrong_steps["error_category"] == "evidence_error"


def test_clean_uncertain_is_safe_but_fail_is_false_positive():
    case = next(case for case in _approved_cases() if case["gold"]["kind"] == "clean")
    uncertain = _score(case, _episode("uncertain"))
    assert uncertain["field_correctness"]["clean_safe"]
    assert not uncertain["field_correctness"]["verdict_exact_gold"]
    assert uncertain["error_category"] == "clean_uncertain"

    finding = {"defect_id": "kinematics.joint_axis", "joint_id": case["public"]["joint_id"],
               "parts": [case["public"]["moving_part"]], "evidence_steps": [2]}
    false_positive = _score(case, _episode("fail", [finding]))
    assert not false_positive["field_correctness"]["clean_safe"]
    assert false_positive["error_category"] == "clean_false_positive"


def test_no_private_answer_is_part_of_public_review_rows():
    for case in _approved_cases():
        public = json.dumps(case["public"], sort_keys=True)
        assert "defect_id" not in public and "parameters" not in public and "seed" not in public
