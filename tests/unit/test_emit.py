from turnitover.assets.emit import emit_program
from turnitover.core.program import ObjectProgram


def test_emit_deterministic(toy_spec):
    assert emit_program(toy_spec) == emit_program(toy_spec)
    assert ObjectProgram.from_spec(toy_spec).sha == ObjectProgram.from_spec(toy_spec).sha


def test_emit_structure(toy_spec):
    src = emit_program(toy_spec)
    assert "export default function createObject(THREE" in src
    for p in toy_spec.parts:
        assert f"g.name = {p.id!r};" in src
    for j in toy_spec.joints:
        assert f"pivot.name = {'joint:' + j.id!r};" in src
    assert "return { root, joints, dispose };" in src


def test_replace_changes_source(toy_spec):
    moved = toy_spec.replace_part("door", origin=(0.1, 0.1, 0.1))
    assert emit_program(moved) != emit_program(toy_spec)
