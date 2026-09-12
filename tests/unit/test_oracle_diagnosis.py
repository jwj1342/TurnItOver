import json

from turnitover.oracle.diagnosis import paired_example


def test_unexposed_truth_does_not_become_finding_or_clean_pass():
    context = dict(task="verify", parts=["door"], joints=["hinge"], views=["front"],
                   runtime_properties=[], action_types=["request_view"], max_triangles=5000, max_draw_calls=32)
    annotation = dict(id="case", labels=[dict(defect_id="kinematics.joint_axis", parts=["door"], severity=.5)],
                      grid=[dict(view="front", joint=None, detent=None)], exposures=[[]],
                      trajectories={"oracle": dict(history=[dict(step=0, action=dict(type="request_view", view_id="front"))], witness_indices=[0])})
    row = paired_example(annotation, context, "oracle", ["candidate.png"], ["reference.png"])
    verdict = json.loads(row["messages"][1]["content"])["verdict"]
    assert verdict["status"] == "uncertain"
    assert not verdict["findings"]
    assert "severity" not in row["messages"][0]["content"] .split("experiment supplies")[-1]
    annotation["exposures"] = [[0]]
    row = paired_example(annotation, context, "oracle", ["candidate.png"], ["reference.png"])
    verdict = json.loads(row["messages"][1]["content"])["verdict"]
    assert verdict["status"] == "fail"
    assert verdict["findings"][0]["evidence_steps"] == [0]
    assert row["images"] == ["reference.png", "candidate.png"]
    annotation["labels"] = []
    verdict = json.loads(paired_example(annotation, context, "oracle", ["c"], ["r"])["messages"][1]["content"])["verdict"]
    assert verdict["status"] == "uncertain"
