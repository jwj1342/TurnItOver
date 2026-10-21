"""Mesh names must not become logical part handles in the Program ABI."""
import pytest

from turnitover.api_pipeline.fixture import reference_program
from turnitover.api_pipeline.scoring import quality
from turnitover.core.program import ObjectProgram
from turnitover.render.session import ObservationSession


pytestmark = pytest.mark.browser


def test_mesh_names_do_not_change_group_part_handles_or_quality(render_config, views):
    reference = reference_program()
    assert "#mesh" in reference.source
    candidate = ObjectProgram(reference.source.replace("#mesh", "_mesh"))
    assert candidate.sha != reference.sha

    with ObservationSession(render_config, views) as session:
        info = session.load(candidate, framing=None)
        assert set(info.parts) == {"base", "drawer", "door"}
        assert set(session.export_geometry()) == {"base", "drawer", "door"}

    result = quality(candidate, reference, render_config, views)
    assert result["status"] == "complete"
    assert result["code_correct"] is True
    assert all(check["passed"] for check in result["checks"])
