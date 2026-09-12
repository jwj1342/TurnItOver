import pytest
import dataclasses

from tests.unit.test_urdf import fixture_model
from turnitover.assets.audit_urdf import audit_model

pytestmark = pytest.mark.browser


def test_browser_matches_independent_urdf_fk(session_factory, tmp_path):
    model = fixture_model(tmp_path)
    result = audit_model(session_factory, model, model.spec("fixture", "fixture"))
    assert result["passed"]
    assert len(result["states"]) == 10
    assert result["max_vertex_error_m"] < 1e-5


def test_audit_rejects_topology_change_with_unchanged_vertices(session_factory, tmp_path):
    model = fixture_model(tmp_path)
    spec = model.spec("fixture", "fixture")
    mesh = spec.part("arm").mesh
    mesh = dataclasses.replace(mesh, faces=tuple((f[2], f[1], f[0]) for f in mesh.faces))
    corrupted = spec.replace_part("arm", mesh=mesh)
    with pytest.raises(ValueError, match="topology changed"):
        audit_model(session_factory, model, corrupted)
