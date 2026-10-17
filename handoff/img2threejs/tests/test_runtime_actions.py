from src.runtime.actions import ViewAction, parse_action, pointer_motion


def test_parse_action() -> None:
    assert parse_action("orbit_left") is ViewAction.ORBIT_LEFT
    assert parse_action(" ZOOM_IN ") is ViewAction.ZOOM_IN


def test_orbit_actions_map_to_drag() -> None:
    left = pointer_motion(ViewAction.ORBIT_LEFT, width=400, height=200)
    up = pointer_motion(ViewAction.ORBIT_UP, width=400, height=200)
    assert left.dx < 0 and left.dy == 0
    assert up.dy < 0 and up.dx == 0


def test_zoom_actions_map_to_wheel() -> None:
    assert pointer_motion(ViewAction.ZOOM_IN, width=400, height=200).wheel_y < 0
    assert pointer_motion(ViewAction.ZOOM_OUT, width=400, height=200).wheel_y > 0
