import json

from PIL import Image

from scripts.prepare_level0_axis_review import AXIS_DEG, DEFAULT_OUT, SELECTIONS, opaque_id


def test_level0_selection_is_five_and_axis_only():
    assert len(SELECTIONS) == 5
    assert AXIS_DEG == 90.0
    assert len({row["asset_id"] for row in SELECTIONS}) == 5


def test_opaque_ids_do_not_reveal_variant_kind():
    clean = opaque_id("PM_1", "clean", None)
    axis = opaque_id("PM_1", "joint_axis", 7)
    assert clean != axis
    assert all(word not in value for value in (clean, axis) for word in ("clean", "axis", "joint"))


def test_generated_level0_bundle_is_balanced_pending_and_checked():
    public = [json.loads(line) for line in (DEFAULT_OUT / "review_manifest.jsonl").read_text().splitlines()]
    gold = [json.loads(line) for line in (DEFAULT_OUT / "private/gold.jsonl").read_text().splitlines()]
    checks = {row["case_id"]: row for row in map(json.loads, (DEFAULT_OUT / "private/checks.jsonl").read_text().splitlines())}
    validation = json.loads((DEFAULT_OUT / "validation.json").read_text())
    assert validation["assets"] == 5 and validation["cases"] == 10
    assert validation["part_offset_cases"] == validation["model_calls"] == validation["sft_training_runs"] == 0
    assert {row["status"] for row in public} == {"pending_human_review"}
    assert {row["kind"] for row in gold} == {"clean", "joint_axis"}
    assert sum(row["kind"] == "clean" for row in gold) == 5
    assert all(row["training_eligibility"] == "pending_human_review" for row in gold)
    assert all(not ({"kind", "defect_id", "joint_id", "parameters"} & row.keys()) for row in public)
    assert all(Image.open(DEFAULT_OUT / row["board"]).size == (912, 630) for row in public)
    assert all(Image.open(DEFAULT_OUT / row["input_preview"]).size == (542, 542) for row in public)
    assert {row["case_id"]: checks[row["case_id"]]["code_correct"] for row in gold} == {
        row["case_id"]: row["kind"] == "clean" for row in gold
    }
