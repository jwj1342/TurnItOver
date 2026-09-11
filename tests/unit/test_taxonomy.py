from pathlib import Path

from turnitover.corruptions import registered_ids
from turnitover.taxonomy import Layer, Visibility, load_taxonomy
from turnitover.taxonomy.render_docs import render_markdown

ROOT = Path(__file__).resolve().parents[2]


def test_taxonomy_integrity():
    tax = load_taxonomy()
    assert len(tax.defects) == 29
    assert len(set(tax.ids)) == 29
    for d in tax.defects:
        assert isinstance(d.layer, Layer) and isinstance(d.visibility, Visibility)
        assert d.label_form in tax.label_forms


def test_kinematics_all_animation_required():
    tax = load_taxonomy()
    kin = [d for d in tax.defects if d.layer is Layer.KINEMATICS]
    assert len(kin) == 5
    assert all(d.visibility is Visibility.ANIMATION_REQUIRED for d in kin)


def test_implemented_matches_registry():
    tax = load_taxonomy()
    assert set(tax.implemented_ids()) == set(registered_ids())


def test_docs_not_drifted():
    expected = render_markdown(load_taxonomy())
    actual = (ROOT / "docs/taxonomy.md").read_text(encoding="utf-8")
    assert actual == expected, "run: python -m turnitover render-docs"
