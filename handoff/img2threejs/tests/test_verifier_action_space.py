from pathlib import Path

import pytest

from src.verifier import (
    PoseBrowserRuntime,
    PoseGridActionSpace,
    RelativeDiscreteActionSpace,
    VerifierEpisode,
    load_action_space,
)


def test_v1_relative_action_space_resolves_semantic_action() -> None:
    space = RelativeDiscreteActionSpace(orbit_drag_fraction=0.3, zoom_wheel_delta=500)
    action = space.resolve("orbit_right")
    assert action.type == "relative"
    assert action.name == "orbit_right"
    assert action.params["orbit_drag_fraction"] == 0.3
    assert action.params["zoom_wheel_delta"] == 500


def test_v2_pose_grid_loads_and_rejects_unknown_view() -> None:
    space = load_action_space("pose_grid_v2", Path("configs/view_pose_grid_v2.json"))
    assert isinstance(space, PoseGridActionSpace)
    action = space.resolve("goto_pose", view_id="az090_el030_d100")
    assert action.type == "goto_pose"
    assert action.params["azimuth_deg"] == 90
    assert action.params["elevation_deg"] == 30
    assert action.params["distance_scale"] == 1.0

    with pytest.raises(ValueError, match="unknown V2 pose"):
        space.resolve("goto_pose", view_id="missing")


def test_v2_runtime_applies_exact_pose_and_records_metadata(
    browser_session,
    fixture_dir: Path,
    tmp_path: Path,
) -> None:
    html = (fixture_dir / "pose_harness_v2.html").read_text(encoding="utf-8")
    space = load_action_space("pose_grid_v2", Path("configs/view_pose_grid_v2.json"))
    assert isinstance(space, PoseGridActionSpace)
    runtime = PoseBrowserRuntime(browser_session, space)
    episode = VerifierEpisode(
        runtime,
        html,
        tmp_path / "v2",
        action_budget=2,
        action_space=space,
    )

    initial = episode.start()
    assert initial.action == "initial"
    assert initial.runtime_metadata["pose"] == {
        "azimuth_deg": 0,
        "elevation_deg": 30,
        "distance_scale": 1,
    }

    observed = episode.act("goto_pose", target_view_id="az090_el030_d100")
    assert observed.action == "goto_pose"
    assert observed.action_payload["params"]["view_id"] == "az090_el030_d100"
    assert observed.runtime_metadata["pose"] == {
        "azimuth_deg": 90,
        "elevation_deg": 30,
        "distance_scale": 1,
    }
    assert Path(observed.screenshot_path).is_file()

    result = episode.finish("Right-side evidence is sufficient.", [observed.view_id])
    assert result.action_space["name"] == "pose_grid_v2"


def test_v2_runtime_refuses_html_without_harness(
    browser_session,
    fixture_dir: Path,
    tmp_path: Path,
) -> None:
    html = (fixture_dir / "valid_threejs.html").read_text(encoding="utf-8")
    space = load_action_space("pose_grid_v2", Path("configs/view_pose_grid_v2.json"))
    assert isinstance(space, PoseGridActionSpace)
    runtime = PoseBrowserRuntime(browser_session, space)
    episode = VerifierEpisode(
        runtime,
        html,
        tmp_path / "v2-no-harness",
        action_budget=1,
        action_space=space,
    )

    with pytest.raises(RuntimeError, match="requires deterministic camera harness"):
        episode.start()
