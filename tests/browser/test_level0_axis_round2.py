"""Audit all new v4 sources through their recorded native URDF locations."""
import json
from pathlib import Path

import pytest

from scripts.prepare_level0_axis_v2 import DEFAULT_OUT, read_rows
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.catalog import CatalogSource
from turnitover.assets.urdf import UrdfModel

pytestmark = pytest.mark.browser
V3 = DEFAULT_OUT.with_name('partnet-mobility-level0-axis-v3')
V4 = DEFAULT_OUT.with_name('partnet-mobility-level0-axis-v4')


def test_new_v4_sources_match_independent_native_urdf_fk(session_factory):
    prior_ids = {g['asset_id'] for g in read_rows(V3 / 'private/gold.jsonl')}
    new_ids = {g['asset_id'] for g in read_rows(V4 / 'private/gold.jsonl')} - prior_ids
    screened = json.loads((V4 / 'private/expansion_screening.json').read_text())['assets']
    selected = {a['asset_id'] for a in screened if a['status'] == 'automatic_pass'} - prior_ids
    assert selected == new_ids
    assert selected
    manifest = json.loads((V4 / 'manifest.json').read_text())
    catalog = CatalogSource(str(V4 / manifest['source_catalog']))
    for aid in sorted(selected):
        source_urdf = Path(catalog.metadata(aid)['source_urdf'])
        assert source_urdf.is_absolute() and source_urdf.is_file()
        result = audit_model(session_factory, UrdfModel(source_urdf), catalog.get(aid))
        assert result['passed'], (aid, result['max_vertex_error_m'])
