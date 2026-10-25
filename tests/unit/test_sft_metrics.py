"""Synthetic evaluation regression cases; no model, GPU or private corpus."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from turnitover.oracle.sft_contracts import parse_stopping
from turnitover.oracle.sft_metrics import object_bootstrap, public_input_digest, score_prediction, summarize


def fixture_record(status="fail", *, joints=("hinge",), clean=None):
    finding = {"defect_id": "structure.part_offset", "parts": ["door"], "joint_id": None,
               "severity": 1.0, "confidence": 1.0, "evidence_steps": [0],
               "description": "Door offset", "suggested_fix": "Align door"}
    target = {"verdict": {"status": status, "confidence": 1.0,
                          "findings": [finding] if status == "fail" else [],
                          "summary": "Audited target", "limitations": ["Limited views"]}}
    payload = {"context": {"task": "diagnose", "parts": ["door"], "joints": list(joints),
                           "views": [], "runtime_properties": [], "action_types": [],
                           "max_triangles": 100, "max_draw_calls": 10}, "history": [{"step": 0}]}
    return {"id": "a:verdict", "episode_id": "a", "images": ["ref.png"],
            "image_sha256": ["a" * 64], "data_identity": {"source_identity_sha256": "b" * 64},
            "is_clean": clean, "messages": [{"role": "user", "content": "Protocol\n" + json.dumps(payload)},
                                              {"role": "assistant", "content": json.dumps(target)}]}


def test_invalid_predictions_are_kept_and_miss_gold_findings():
    record = fixture_record()
    valid = score_prediction(record, record["messages"][1]["content"])
    invalid = score_prediction(record, "not json")
    report = summarize([valid, invalid])
    assert report["n"] == 2
    assert report["valid_output_rate"] == .5
    assert report["defect_per_class"]["structure.part_offset"]["f1"] == pytest.approx(2 / 3)
    assert report["finding_micro"]["fn"] == 1
    assert report["evidence_support_rate"] == 1
    assert report["joint_localization_micro"] is None


@pytest.mark.parametrize("field,value", [("parts", ["invented"]), ("evidence_steps", [99])])
def test_illegal_parts_or_unobserved_citations_invalidate_whole_prediction(field, value):
    record = fixture_record()
    prediction = json.loads(record["messages"][1]["content"])
    prediction["verdict"]["findings"][0][field] = value
    row = score_prediction(record, json.dumps(prediction))
    assert not row["valid"]
    assert row["error"]
    assert summarize([row])["finding_micro"]["fn"] == 1


def test_observed_but_wrong_evidence_not_supported():
    record = fixture_record()
    payload = json.loads(record["messages"][0]["content"].split("\n")[-1])
    payload["history"].append({"step": 1})
    record["messages"][0]["content"] = "Protocol\n" + json.dumps(payload)
    prediction = json.loads(record["messages"][1]["content"])
    prediction["verdict"]["findings"][0]["evidence_steps"] = [1]
    row = score_prediction(record, json.dumps(prediction))
    assert row["valid"]
    assert summarize([row])["evidence_support_rate"] == 0


def test_false_pass_and_clean_false_positive_have_explicit_denominators():
    record = fixture_record(status="uncertain", clean=True)
    fail = fixture_record()["messages"][1]["content"]
    passed = fixture_record(status="pass")["messages"][1]["content"]
    report = summarize([score_prediction(record, fail), score_prediction(record, passed)])
    assert report["clean_cases"] == 2
    assert report["clean_false_positive_rate"] == .5
    assert report["false_pass_rate"] == .5
    record.pop("is_clean")
    assert summarize([score_prediction(record, fail)])["clean_false_positive_rate"] is None


def test_public_digest_ignores_supervision_targets_but_catches_prompt_change():
    record = fixture_record()
    other = copy.deepcopy(record)
    other["messages"][1]["content"] = "ordinary gold"
    assert public_input_digest([record]) == public_input_digest([other])
    other["messages"][0]["content"] += "changed"
    assert public_input_digest([record]) != public_input_digest([other])


def test_bootstrap_clusters_corruptions_of_original_object():
    record = fixture_record()
    rows = [score_prediction(record, record["messages"][1]["content"]) for _ in range(3)]
    assert object_bootstrap(rows, samples=20)["ci95"] is None
    other = copy.deepcopy(rows[0])
    other["object_id"] = "different-original"
    rows.append(other)
    report = object_bootstrap(rows, samples=20, seed=4)
    assert report["independent_objects"] == 2
    assert report["unit"] == "original_source_object"
    assert report["ci95"] == [1, 1]


def test_stopping_is_separate_and_invalid_rows_remain():
    record = fixture_record()
    target = {"stop_decision": {"decision": "continue", "outcome": None, "reason": "Need more views"}}
    record["messages"][1]["content"] = json.dumps(target)
    rows = [score_prediction(record, json.dumps(target), "stopping"), score_prediction(record, "broken", "stopping")]
    assert summarize(rows, "stopping")["decision_accuracy"] == .5
    with pytest.raises(ValueError):
        summarize(rows)
    with pytest.raises(ValueError):
        parse_stopping({"stop_decision": {"decision": "continue", "outcome": "pass", "reason": "bad"}})


def test_comparison_rejects_mismatched_inputs_and_incomplete_runs():
    path = Path(__file__).resolve().parents[2] / "scripts" / "compare_qwen_sft.py"
    spec = importlib.util.spec_from_file_location("compare_sft", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = {"effective_generation_config_sha256": "generation", "defect_ids": ["structure.part_offset"], "processor_sha256": "p", "device": "cpu", "version": "v1", "task": "diagnosis", "split": "validation", "dataset_version": "synthetic-v1",
                "dataset_manifest_sha256": "a", "annotations_sha256": "ann", "split_sha256": "b", "public_input_sha256": "c",
                "max_new_tokens": 20, "do_sample": False, "seed": 0, "n": 2, "independent_objects": 2}
    report = {"status": "complete", "protocol": protocol, "n": 2, "independent_objects": 2}
    assert module.compare({"base": report, "evidence": copy.deepcopy(report)})["status"] == "complete"
    other = copy.deepcopy(report)
    other["protocol"]["public_input_sha256"] = "changed"
    with pytest.raises(ValueError, match="protocol mismatch"):
        module.compare({"base": report, "sft": other})
    for key in ("effective_generation_config_sha256",):
        changed_generation = copy.deepcopy(report)
        changed_generation["protocol"][key] = "changed"
        with pytest.raises(ValueError, match="protocol mismatch"):
            module.compare({"base": report, "sft": changed_generation})
    other_model = copy.deepcopy(report)
    other_model["model_config_sha256"] = "other-size"
    assert module.compare({"base": report, "other_size": other_model})["status"] == "complete"
    other = copy.deepcopy(report)
    other["status"] = "incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        module.compare({"base": report, "sft": other})


def test_macro_labels_fixed_across_predictions_and_joint_none_not_scored():
    record = fixture_record(joints=("a", "b"))
    valid = score_prediction(record, record["messages"][1]["content"])
    invalid = score_prediction(record, "bad")
    first, second = summarize([valid]), summarize([invalid])
    assert first["defect_labels"] == second["defect_labels"]
    assert first["joint_localization_micro"]["tp"] == 0
    assert first["joint_localization_micro"]["f1"] is None
    assert first["defect_macro_f1"] == 1 / len(first["defect_labels"])
    record["is_clean"] = "true"
    assert summarize([score_prediction(record, record["messages"][1]["content"])])["clean_cases"] == 0


def test_given_joint_not_an_open_localization_success():
    record = fixture_record(joints=("hinge",))
    target = json.loads(record["messages"][1]["content"])
    target["verdict"]["findings"][0].update(defect_id="kinematics.joint_axis", joint_id="hinge")
    record["messages"][1]["content"] = json.dumps(target)
    row = score_prediction(record, json.dumps(target))
    assert row["valid"]
    assert summarize([row])["joint_localization_micro"] is None
