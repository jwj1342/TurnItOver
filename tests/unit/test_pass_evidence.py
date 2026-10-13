import io

from PIL import Image

from turnitover.oracle.pass_evidence import (
    FRONT_REST_SCOPE,
    RUNTIME_BUDGET_SCOPE,
    assess_single_reference_pass,
)


def _png(color):
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(stream, format="PNG")
    return stream.getvalue()


def _payload(task, history, images):
    return {
        "context": {"task": task, "max_triangles": 100, "max_draw_calls": 10},
        "history": history,
        "image_order": images,
    }


def test_front_rest_pass_requires_explicit_narrow_scope_and_visible_pixel_match(tmp_path):
    (tmp_path / "reference.png").write_bytes(_png("white"))
    (tmp_path / "candidate.png").write_bytes(_png("white"))
    history = [{"step": 0, "action": {"type": "request_view", "view_id": "front"},
                "image_ref": "candidate.png", "payload": {}}]

    result = assess_single_reference_pass(
        _payload(FRONT_REST_SCOPE, history, ["reference.png", "candidate.png"]), tmp_path)
    assert result["eligible"] and result["scope"] == "front_rest_appearance_only"

    broad = assess_single_reference_pass(
        _payload("Verify the articulated candidate.", history, ["reference.png", "candidate.png"]), tmp_path)
    assert not broad["eligible"] and broad["front_rest_exact_match"]


def test_front_rest_scope_rejects_mismatch_and_post_actuation_view(tmp_path):
    (tmp_path / "reference.png").write_bytes(_png("white"))
    (tmp_path / "candidate.png").write_bytes(_png("black"))
    mismatch = [{"step": 0, "action": {"type": "request_view", "view_id": "front"},
                 "image_ref": "candidate.png", "payload": {}}]
    assert not assess_single_reference_pass(
        _payload(FRONT_REST_SCOPE, mismatch, ["reference.png", "candidate.png"]), tmp_path)["eligible"]

    moved = [{"step": 0, "action": {"type": "actuate_joint", "joint_id": "hinge", "detent": "mid"},
              "image_ref": "candidate.png", "payload": {}}, *mismatch]
    assert not assess_single_reference_pass(
        _payload(FRONT_REST_SCOPE, moved, ["reference.png", "candidate.png"]), tmp_path)["eligible"]


def test_runtime_pass_requires_explicit_scope_and_complete_public_stats(tmp_path):
    under = [{"step": 0, "action": {"type": "query_runtime", "property": "stats"},
              "image_ref": None, "payload": {"triangles_scene": 90, "draw_calls": 8}}]
    result = assess_single_reference_pass(_payload(RUNTIME_BUDGET_SCOPE, under, []), tmp_path)
    assert result["eligible"] and result["scope"] == "runtime_budget_only"

    over = [{**under[0], "payload": {"triangles_scene": 101, "draw_calls": 8}}]
    assert not assess_single_reference_pass(_payload(RUNTIME_BUDGET_SCOPE, over, []), tmp_path)["eligible"]
    malformed = [{**under[0], "payload": {"triangles_scene": -1, "draw_calls": True}}]
    assert not assess_single_reference_pass(
        _payload(RUNTIME_BUDGET_SCOPE, malformed, []), tmp_path)["eligible"]
    assert not assess_single_reference_pass(_payload(RUNTIME_BUDGET_SCOPE, [], []), tmp_path)["eligible"]
