"""Independent native-URDF authenticity checks for newly selected v3 sources."""
import json

import pytest

from scripts.prepare_level0_axis_v2 import DEFAULT_OUT, read_rows
from scripts.prepare_obvious_defect_review import ROOT
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.catalog import CatalogSource
from turnitover.assets.urdf import UrdfModel

pytestmark = pytest.mark.browser
V3 = DEFAULT_OUT.with_name('partnet-mobility-level0-axis-v3')


def test_new_v3_sources_match_independent_native_urdf_fk(session_factory):
    baseline_ids = {g['asset_id'] for g in read_rows(DEFAULT_OUT / 'private/gold.jsonl')}
    new_ids = {g['asset_id'] for g in read_rows(V3 / 'private/gold.jsonl')} - baseline_ids
    screened = json.loads((V3 / 'private/expansion_screening.json').read_text())['assets']
    selected = {a['asset_id'] for a in screened if a['status'] == 'automatic_pass'} - baseline_ids
    assert selected == new_ids
    manifest = json.loads((V3 / 'manifest.json').read_text())
    catalog = CatalogSource(str(V3 / manifest['source_catalog']))
    for aid in sorted(selected):
        model = UrdfModel(ROOT / 'data/partnet-mobility-pilot/extracted' / aid[3:] / 'mobility.urdf')
        result = audit_model(session_factory, model, catalog.get(aid))
        assert result['passed'], (aid, result['max_vertex_error_m'])
