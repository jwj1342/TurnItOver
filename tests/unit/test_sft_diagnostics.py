import json

from scripts.analyze_qwen_episode_diagnostics import summarize, summarize_defect
from scripts.evaluate_qwen_episode_diagnostics import remove_remaining, visible_findings


def test_visible_findings_uses_only_causal_witness_prefix():
    annotation = {
        "labels": [
            {"defect_id": "structure.part_offset", "parts": ["door"], "params": {}},
            {"defect_id": "kinematics.joint_axis", "parts": ["door"],
             "params": {"joint": "hinge"}},
        ],
        "exposures": [[0], [1]],
        "trajectories": {"oracle": {
            "history": [{"step": 0}, {"step": 1}], "witness_indices": [0, 1],
        }},
    }
    assert visible_findings(annotation, 1) == {
        ("structure.part_offset", None, ("door",)): {0},
    }


def test_remove_remaining_preserves_real_history_and_other_payload_fields():
    payload = {"context": {"task": "x"}, "remaining": 4,
               "history": [{"step": 0}], "image_order": ["a", "b"]}
    record = {"messages": [{"role": "user", "content": "protocol\n" + json.dumps(payload)},
                            {"role": "assistant", "content": "{}"}]}
    changed = remove_remaining(record)
    changed_payload = json.loads(changed["messages"][0]["content"].rsplit("\n", 1)[-1])
    assert "remaining" not in changed_payload
    assert changed_payload["history"] == payload["history"]
    assert record["messages"][0]["content"].endswith(json.dumps(payload))


def test_diagnostic_summary_counts_exact_and_unsupported_findings():
    expected = [["structure.part_offset", None, ["door"]]]
    rows = [
        {"decision_kind": "verdict", "predicted_status": "fail", "expected_status": "fail",
         "expected_findings": expected, "predicted_findings": expected, "unsupported_findings": 0},
        {"decision_kind": "verdict", "predicted_status": "fail", "expected_status": "uncertain",
         "expected_findings": [], "predicted_findings": expected, "unsupported_findings": 1},
    ]
    result = summarize(rows)
    assert result["exact_verdict"] == 1
    assert result["findings"] == {"tp": 1, "fp": 1, "fn": 0,
                                  "precision": .5, "recall": 1., "f1": 2 / 3}
    assert result["false_fail"] == 1
    assert result["unsupported_findings"] == 1
    assert summarize_defect(rows, "structure.part_offset") == {
        "tp": 1, "fp": 1, "fn": 0, "precision": .5, "recall": 1., "f1": 2 / 3,
        "expected_cases": 1, "predicted_cases": 2,
    }
