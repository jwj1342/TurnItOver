import pytest

from turnitover.core.program import ObjectProgram
from turnitover.render.gate import run_render_gate


pytestmark = pytest.mark.browser


def test_native_render_gate_accepts_visible_program_and_structures_compile_failure(
    tmp_path, toy_program, render_config, views
):
    accepted = run_render_gate(toy_program, tmp_path / "accepted", render_config, views)
    assert accepted.success and accepted.image_ref == "render.png"
    assert accepted.image_stddev is not None and accepted.image_stddev > 1

    rejected = run_render_gate(ObjectProgram("export default 42"), tmp_path / "rejected", render_config, views)
    assert not rejected.success
    assert rejected.error_type in {"CompileError", "HarnessError"}
