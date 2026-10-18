"""Replay every selected v2 trajectory through Chromium and the real checker."""
import json

import pytest
from PIL import Image
import io

from scripts.finalize_level0_axis_v2 import rows
from scripts.prepare_level0_axis_review import screen
from scripts.prepare_level0_axis_v2 import DEFAULT_OUT
from scripts.level0_expansion_search import inside_camera, render_states
from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, checker_result, render_program
from turnitover.assets.catalog import CatalogSource, decode_spec
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.urdf import UrdfModel
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing
from turnitover.render.views import load_views

pytestmark = pytest.mark.browser


def test_all_selected_v2_pairs_reproduce_native_pixels_and_checker(session_factory):
    gold = rows(DEFAULT_OUT / 'private/gold.jsonl')
    public = {r['case_id']: r for r in rows(DEFAULT_OUT / 'review_manifest.jsonl')}
    manifest = json.loads((DEFAULT_OUT / 'manifest.json').read_text())
    catalog = CatalogSource(str(DEFAULT_OUT / manifest['source_catalog']))
    views = {str(v.id): v for v in load_views(load_generate_config(DEFAULT_CONFIG).views_path)}
    inherited = {a['asset_id']: a for a in json.loads((DEFAULT_OUT / 'private/inherited_screening.json').read_text())['assets']}
    new = {a['asset_id']: a for a in json.loads((DEFAULT_OUT / 'private/screening.json').read_text())['assets']}
    expansion = {a['asset_id']: a for a in json.loads((DEFAULT_OUT / 'private/expansion_screening.json').read_text())['assets']}
    for axis in (g for g in gold if g['kind'] == 'joint_axis'):
        asset = axis['asset_id']
        clean = next(g for g in gold if g['asset_id'] == asset and g['kind'] == 'clean')
        diagnostic = inherited.get(asset)
        if diagnostic is None:
            diagnostic = next(j for j in (expansion.get(asset) or new[asset])['joints'] if 'selected' in j)
        f = diagnostic['normal_framing']
        framing = Framing(center=tuple(f['center']), radius=f['radius'])
        reference = ObjectProgram.from_spec(catalog.get(asset))
        view = public[axis['case_id']]['view']
        def replay(program):
            if 'state_policy' in diagnostic:
                return render_states(session_factory, program, framing, axis['joint_id'], diagnostic['state_values'], [view])
            return render_program(session_factory, program, framing, axis['joint_id'], [view])
        ref = replay(reference)
        for g in (clean, axis):
            program = ObjectProgram.from_spec(decode_spec(json.loads((DEFAULT_OUT / 'private' / f"{g['case_id']}.json").read_text())))
            rendered = replay(program)
            check = checker_result(program, rendered, reference, ref)
            assert check['code_correct'] == (g['kind'] == 'clean')
            if g['kind'] == 'joint_axis':
                assert screen(ref, rendered, view, check)[1] == []
            if 'state_policy' in diagnostic:
                assert inside_camera(rendered, framing, views[view])[0]
            expected_frames = [ref['frames']['rest'][view], ref['frames']['limit'][view],
                               rendered['frames']['rest'][view], rendered['frames']['limit'][view]]
            for path, png in zip(public[g['case_id']]['images'], expected_frames):
                assert Image.open(DEFAULT_OUT / path).convert('RGB').tobytes() == Image.open(io.BytesIO(png)).convert('RGB').tobytes()


def test_expanded_sources_match_independent_native_urdf_fk(session_factory):
    from scripts.prepare_obvious_defect_review import ROOT
    selected = [a for a in json.loads((DEFAULT_OUT / 'private/expansion_screening.json').read_text())['assets']
                if a['status'] == 'automatic_pass']
    manifest = json.loads((DEFAULT_OUT / 'manifest.json').read_text())
    catalog = CatalogSource(str(DEFAULT_OUT / manifest['source_catalog']))
    assert len(selected) >= 21
    for row in selected:
        aid = row['asset_id']
        model = UrdfModel(ROOT / 'data/partnet-mobility-pilot/extracted' / aid[3:] / 'mobility.urdf')
        result = audit_model(session_factory, model, catalog.get(aid))
        assert result['passed'], (aid, result['max_vertex_error_m'])
