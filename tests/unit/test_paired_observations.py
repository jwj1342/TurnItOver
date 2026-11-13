import copy
import json

import pytest

from turnitover.oracle.paired_observations import paired_terminal, user_content
from turnitover.oracle.supervision_validation import public_payload


def paired_record():
    context = {"task": "compare articulation", "parts": ["link_0"], "joints": ["joint_0"],
               "views": ["front"], "runtime_properties": [], "action_types": ["request_view"],
               "max_triangles": 1, "max_draw_calls": 1}
    history = [{"step": step, "state": state,
                "action": {"type": "request_view", "view_id": "front"},
                "reference_image_ref": f"reference_{state}.png", "image_ref": f"candidate_{state}.png"}
               for step, state in enumerate(("zero", "mid", "limit"))]
    labels = [{"defect_id": "kinematics.joint_axis", "parts": ["link_0"],
               "severity": .6, "params": {"joint": "joint_0"}}]
    return paired_terminal("case", context, history, [[], [0], [0]], labels)


def set_payload(record, payload):
    protocol = record["messages"][0]["content"].rsplit("\n", 1)[0]
    record["messages"][0]["content"] = protocol + "\n" + json.dumps(payload)


def test_six_images_are_three_independent_pairs_and_citations_reference_pairs():
    row = paired_record()
    payload = public_payload(row)
    assert row["images"] == [f"{role}_{state}.png" for state in ("zero", "mid", "limit")
                             for role in ("reference", "candidate")]
    assert payload["history"][0]["reference_image_ref"] == "reference_zero.png"
    target = json.loads(row["messages"][1]["content"])
    assert target["verdict"]["findings"][0]["evidence_steps"] == [1, 2]
    assert "not an individual image" in row["messages"][0]["content"]
    assert "Reference images come first" not in row["messages"][0]["content"]


@pytest.mark.parametrize("mutation", ["state", "step", "view", "order", "missing_reference", "format", "unmarked", "private"])
def test_paired_input_rejects_misaligned_or_private_observations(mutation):
    row = paired_record()
    payload = copy.deepcopy(public_payload(row))
    if mutation == "state":
        payload["history"][1]["state"] = "zero"
    elif mutation == "step":
        payload["history"][2]["step"] = 3
    elif mutation == "view":
        payload["history"][1]["action"]["view_id"] = "back"
    elif mutation == "order":
        row["images"][0], row["images"][1] = row["images"][1], row["images"][0]
        payload["image_order"] = row["images"]
    elif mutation == "missing_reference":
        del payload["history"][0]["reference_image_ref"]
    elif mutation == "format":
        payload["observation_format"] = "unknown"
    elif mutation == "unmarked":
        del payload["observation_format"]
    else:
        payload["history"][0]["gold"] = {"defect": "axis"}
    set_payload(row, payload)
    with pytest.raises(ValueError):
        public_payload(row)


def test_shared_user_content_explicitly_labels_each_image_role():
    row = paired_record()
    images = [object() for _ in range(6)]
    content = user_content(row, images)
    assert [part["image"] for part in content if part["type"] == "image"] == images
    for index, state in enumerate(("zero", "mid", "limit")):
        offset = 1 + 4 * index
        assert content[offset]["text"] == f"Observation {index}, state={state}, view=front: reference"
        assert content[offset + 1]["image"] is images[index * 2]
        assert content[offset + 2]["text"] == f"Observation {index}: candidate"
        assert content[offset + 3]["image"] is images[index * 2 + 1]


def test_paired_prefix_images_and_user_text_are_masked_out_of_loss():
    torch = pytest.importorskip("torch")
    from turnitover.oracle.training import tokenize_example

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert len([part for part in messages[0]["content"] if part["type"] == "image"]) == 6
            length = 30 if len(messages) == 1 else 35
            return {"input_ids": torch.arange(length).reshape(1, -1),
                    "attention_mask": torch.ones((1, length), dtype=torch.long)}

    batch = tokenize_example(Processor(), paired_record(), [object() for _ in range(6)])
    assert batch["labels"].tolist() == [[-100] * 30 + list(range(30, 35))]


def paired_annotation():
    return {"labels": [{"defect_id": "kinematics.joint_axis", "parts": ["link_0"],
                        "severity": .6, "params": {"joint": "joint_0"}}]}


def test_paired_evidence_audits_citations_against_private_pair_support():
    from turnitover.oracle.sft_data import validate_target
    row = paired_record()
    assert row["evidence"]["pair_label_indices"] == [[], [0], [0]]
    validate_target(row, "diagnosis", "evidence", paired_annotation())
    target = json.loads(row["messages"][1]["content"])
    # Step 0 is a legal index but exposes no defect; format validation alone is insufficient.
    target["verdict"]["findings"][0]["evidence_steps"] = [0, 1, 2]
    row["messages"][1]["content"] = json.dumps(target)
    with pytest.raises(ValueError, match="supporting pairs"):
        validate_target(row, "diagnosis", "evidence", paired_annotation())
    # Ordinary intentionally cites all observed pairs regardless of exposure.
    validate_target(row, "diagnosis", "ordinary", paired_annotation())


@pytest.mark.parametrize("mutation", ["missing", "short", "unknown_label", "duplicate", "boolean", "coverage"])
def test_invalid_private_pair_support_is_rejected(mutation):
    from turnitover.oracle.sft_data import validate_target
    row = paired_record()
    evidence = row["evidence"]
    if mutation == "missing":
        del evidence["pair_label_indices"]
    elif mutation == "short":
        evidence["pair_label_indices"] = [[0]]
    elif mutation == "unknown_label":
        evidence["pair_label_indices"][0] = [1]
    elif mutation == "duplicate":
        evidence["pair_label_indices"][0] = [0, 0]
    elif mutation == "boolean":
        evidence["pair_label_indices"][0] = [False]
    else:
        evidence["covered_label_indices"] = []
    with pytest.raises(ValueError, match="exposure"):
        validate_target(row, "diagnosis", "evidence", paired_annotation())


def test_pair_support_is_specific_to_each_defect_not_just_covered_union():
    from turnitover.oracle.sft_data import validate_target
    row = paired_record()
    payload = public_payload(row)
    labels = paired_annotation()["labels"] + [{"defect_id": "structure.part_offset", "parts": ["link_0"],
                                              "severity": .6, "params": {}}]
    row = paired_terminal("case", payload["context"], payload["history"], [[], [0], [1]], labels)
    validate_target(row, "diagnosis", "evidence", {"labels": labels})
    target = json.loads(row["messages"][1]["content"])
    # Both indices support some defect, but swapping them must not pass.
    target["verdict"]["findings"][0]["evidence_steps"] = [2]
    target["verdict"]["findings"][1]["evidence_steps"] = [1]
    row["messages"][1]["content"] = json.dumps(target)
    with pytest.raises(ValueError, match="supporting pairs"):
        validate_target(row, "diagnosis", "evidence", {"labels": labels})
