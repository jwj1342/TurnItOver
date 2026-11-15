from pathlib import Path

from turnitover.corruptions import registered_ids
from turnitover.taxonomy import load_taxonomy
from turnitover.taxonomy.render_docs import render_markdown

ROOT = Path(__file__).resolve().parents[2]


def test_implemented_matches_registry():
    tax = load_taxonomy()
    assert set(tax.implemented_ids()) == set(registered_ids())


def test_docs_not_drifted():
    expected = render_markdown(load_taxonomy())
    actual = (ROOT / "docs/taxonomy.md").read_text(encoding="utf-8")
    assert actual == expected, "run: python -m turnitover render-docs"
