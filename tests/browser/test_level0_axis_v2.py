"""Replay every selected v2 trajectory through Chromium and the real checker."""
import json

import pytest
from PIL import Image
import io

from scripts.finalize_level0_axis_v2 import rows
from scripts.prepare_level0_axis_review import screen
from scripts.prepare_level0_axis_v2 import DEFAULT_OUT
from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, checker_result, render_program
from turnitover.assets.catalog import CatalogSource, decode_spec
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing

pytestmark = pytest.mark.browser


def test_all_selected_v2_pairs_reproduce_native_pixels_and_checker(session_factory):
    gold = rows(DEFAULT_OUT / 'private/gold.jsonl')
    public = {r['case_id']: r for r in rows(DEFAULT_OUT / 'review_manifest.jsonl')}
    catalog = CatalogSource(load_generate_config(DEFAULT_CONFIG).asset_source['manifest'])
    inherited = {a['asset_id']: a for a in json.loads((DEFAULT_OUT / 'private/inherited_screening.json').read_text())['assets']}
    new = {a['asset_id']: a for a in json.loads((DEFAULT_OUT / 'private/screening.json').read_text())['assets']}
    for axis in (g for g in gold if g['kind'] == 'joint_axis'):
        asset = axis['asset_id']
        clean = next(g for g in gold if g['asset_id'] == asset and g['kind'] == 'clean')
        diagnostic = inherited.get(asset)
        if diagnostic is None:
            diagnostic = next(j for j in new[asset]['joints'] if 'selected' in j)
        f = diagnostic['normal_framing']
        framing = Framing(center=tuple(f['center']), radius=f['radius'])
        reference = ObjectProgram.from_spec(catalog.get(asset))
        view = public[axis['case_id']]['view']
        ref = render_program(session_factory, reference, framing, axis['joint_id'], [view])
        for g in (clean, axis):
            program = ObjectProgram.from_spec(decode_spec(json.loads((DEFAULT_OUT / 'private' / f"{g['case_id']}.json").read_text())))
            rendered = render_program(session_factory, program, framing, axis['joint_id'], [view])
            check = checker_result(program, rendered, reference, ref)
            assert check['code_correct'] == (g['kind'] == 'clean')
            if g['kind'] == 'joint_axis':
                assert screen(ref, rendered, view, check)[1] == []
            expected_frames = [ref['frames']['rest'][view], ref['frames']['limit'][view],
                               rendered['frames']['rest'][view], rendered['frames']['limit'][view]]
            for path, png in zip(public[g['case_id']]['images'], expected_frames):
                assert Image.open(DEFAULT_OUT / path).convert('RGB').tobytes() == Image.open(io.BytesIO(png)).convert('RGB').tobytes()
