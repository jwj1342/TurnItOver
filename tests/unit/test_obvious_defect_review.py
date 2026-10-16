from scripts.prepare_obvious_defect_review import ROOT, blind_id, image_metrics

from PIL import Image
import io
import json


def _png(color):
    out = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(out, format="PNG")
    return out.getvalue()


def test_blind_ids_are_stable_and_do_not_reveal_labels():
    values = [blind_id("PM_1", kind, 7) for kind in ("clean", "joint_axis", "part_offset")]
    assert values == [blind_id("PM_1", kind, 7) for kind in ("clean", "joint_axis", "part_offset")]
    assert len(set(values)) == 3
    assert all(label not in value for value in values for label in ("clean", "axis", "offset"))


def test_image_metrics_distinguish_equal_and_large_changes():
    black, white = _png("black"), _png("white")
    assert image_metrics(black, black) == {
        "changed_fraction": 0.0, "core_changed_fraction": 0.0, "largest_core_region_px": 0}
    changed = image_metrics(white, black)
    assert changed["changed_fraction"] == 1.0
    assert changed["core_changed_fraction"] > 0.5
    assert changed["largest_core_region_px"] > 100


def test_generated_bundle_is_blind_pending_and_checker_confirmed():
    root = ROOT / "docs/evidence/partnet-mobility-obvious-defects-v1"
    manifest = json.loads((root / "manifest.json").read_text())
    public = [json.loads(line) for line in (root / "review_manifest.jsonl").read_text().splitlines()]
    gold = [json.loads(line) for line in (root / "private/gold.jsonl").read_text().splitlines()]
    checks = [json.loads(line) for line in (root / "private/checks.jsonl").read_text().splitlines()]
    validation = json.loads((root / "validation.json").read_text())

    assert manifest["cases"] == 9 and manifest["assets"] == 3
    assert manifest["student_image_size"] == [256, 256]
    assert manifest["model_calls"] == manifest["sft_training_runs"] == manifest["test_rows_touched"] == 0
    assert {row["status"] for row in public} == {"pending_human_review"}
    assert all(not ({"kind", "defect_id", "parameters"} & row.keys()) for row in public)
    assert len({row["case_id"] for row in public}) == 9
    assert all(Image.open(root / row["board"]).size == (920, 648) for row in public)

    by_asset = {}
    for row in gold:
        asset_id = next(item["asset_id"] for item in public if item["case_id"] == row["case_id"])
        by_asset.setdefault(asset_id, set()).add(row["kind"])
        assert row["training_eligibility"] == "pending_human_review"
    assert all(kinds == {"clean", "joint_axis", "part_offset"} for kinds in by_asset.values())
    expected = {row["case_id"]: row["kind"] == "clean" for row in gold}
    assert {row["case_id"]: row["code_correct"] for row in checks} == expected
    assert validation == {
        "status": "passed", "cases": 9, "train_assets": 3, "pending_human_review": 9,
        "clean_checker_passed": 3, "single_defect_checker_rejected": 6,
        "native_image_size": [256, 256], "public_answer_fields": [],
        "test_rows_touched": 0, "model_calls": 0, "sft_training_runs": 0}
