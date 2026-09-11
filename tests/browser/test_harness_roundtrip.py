import io

import numpy as np
import pytest
from PIL import Image

from turnitover.core.actions import Detent, RuntimeProperty

pytestmark = pytest.mark.browser


def test_load_view_actuate_export(session_factory, toy_program):
    s = session_factory
    info = s.load(toy_program, framing=None)
    assert len(info.parts) == 9 and set(info.joints) == {"drawer_0", "drawer_1", "door"}
    assert info.framing.radius > 0

    v = s.request_view("front")
    img = Image.open(io.BytesIO(v.png))
    assert img.size == (256, 256)

    rest = s.export_geometry()
    assert set(rest) == set(info.parts)
    stats = s.query_runtime(RuntimeProperty.STATS)
    assert stats["mesh_count"] == 9 and stats["triangles_scene"] == 12 * 9
    assert stats["draw_calls"] >= 1

    a = s.actuate("drawer_0", Detent.LIMIT)
    assert abs(a.value - 0.35) < 1e-6
    moved = s.export_geometry()
    dz = moved["drawer_0"].vertices.mean(axis=0) - rest["drawer_0"].vertices.mean(axis=0)
    assert np.allclose(dz, [0, 0, 0.35], atol=1e-4)
    assert np.allclose(moved["body_back"].vertices, rest["body_back"].vertices)

    delta = s.query_runtime(RuntimeProperty.STATE_DELTA)
    assert delta["since"].startswith("actuate:drawer_0")
    assert abs(delta["parts"]["drawer_0"]["aabb_min_delta"][2] - 0.35) < 1e-4

    h = s.query_runtime(RuntimeProperty.HIERARCHY)
    names = {c["name"] for c in h["root"]["children"]}
    assert "joint:drawer_0" in names and "body_back" in names
    s.dispose()


def test_compile_error_surfaces(session_factory, toy_program):
    from turnitover.core.program import ObjectProgram
    from turnitover.render.session import CompileError

    with pytest.raises(CompileError):
        session_factory.load(ObjectProgram(source="export default function ( {"), framing=None)


def test_abi_violation_is_harness_error(session_factory):
    from turnitover.core.program import ObjectProgram
    from turnitover.render.session import HarnessError

    with pytest.raises(HarnessError) as e:
        session_factory.load(ObjectProgram(source="export default 42;"), framing=None)
    assert e.value.stage == "validate"
