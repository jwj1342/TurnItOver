"""Reproduce every native v2 image from its frozen programs, states and cameras."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image

from scripts.level0_expansion_search import inside_camera, render_states
from scripts.prepare_level0_axis_review import board, pair_preview, screen
from scripts.prepare_level0_axis_v2 import DEFAULT_OUT, read_rows
from scripts.prepare_obvious_defect_review import DEFAULT_CONFIG, checker_result, render_program, sha256_path
from turnitover.assets.catalog import decode_spec
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views


def replay(dataset: Path, out: Path):
    if out.exists():
        raise ValueError('Replay output must not exist')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha256_path(dataset / name) != digest:
            raise ValueError(f'Input SHA mismatch: {name}')
    gold = read_rows(dataset / 'private/gold.jsonl')
    public = {r['case_id']: r for r in read_rows(dataset / 'review_manifest.jsonl')}
    inherited = {a['asset_id']: a for a in json.loads((dataset / 'private/inherited_screening.json').read_text())['assets']}
    normal = {a['asset_id']: a for a in json.loads((dataset / 'private/screening.json').read_text())['assets']}
    expanded_path = dataset / 'private/expansion_screening.json'
    expanded = ({a['asset_id']: a for a in json.loads(expanded_path.read_text())['assets']}
                if expanded_path.exists() else {})
    cfg = load_generate_config(DEFAULT_CONFIG)
    assert (cfg.render.width, cfg.render.height) == (256, 256)
    views = load_views(cfg.views_path)
    definitions = {str(v.id): v for v in views}
    out.mkdir(parents=True)
    checked = []
    with ObservationSession(cfg.render, views) as session:
        for axis in (g for g in gold if g['kind'] == 'joint_axis'):
            aid = axis['asset_id']
            clean = next(g for g in gold if g['asset_id'] == aid and g['kind'] == 'clean')
            diagnostic = inherited.get(aid)
            if diagnostic is None:
                diagnostic = next(j for j in (expanded.get(aid) or normal[aid])['joints'] if 'selected' in j)
            f = diagnostic['normal_framing']
            framing = Framing(center=tuple(f['center']), radius=f['radius'])
            view = public[axis['case_id']]['view']
            def program(row):
                result = ObjectProgram.from_spec(decode_spec(json.loads((dataset / 'private' / f"{row['case_id']}.json").read_text())))
                assert result.sha == row['candidate_sha']
                return result
            def render(p):
                if 'state_policy' in diagnostic:
                    return render_states(session, p, framing, axis['joint_id'], diagnostic['state_values'], [view])
                return render_program(session, p, framing, axis['joint_id'], [view])
            reference = program(clean)
            ref = render(reference)
            for row in (clean, axis):
                cid = row['case_id']
                candidate = program(row)
                rendered = ref if row['kind'] == 'clean' else render(candidate)
                check = checker_result(candidate, rendered, reference, ref)
                assert check['code_correct'] == (row['kind'] == 'clean'), cid
                if row['kind'] == 'joint_axis':
                    assert screen(ref, rendered, view, check)[1] == [], cid
                if 'state_policy' in diagnostic:
                    assert inside_camera(rendered, framing, definitions[view])[0], cid
                board(out / public[cid]['board'], cid, ref, rendered, view)
                pair_preview(out / public[cid]['input_preview'], ref, rendered, view)
                frames = (ref['frames']['rest'][view], ref['frames']['limit'][view],
                          rendered['frames']['rest'][view], rendered['frames']['limit'][view])
                for path, png in zip(public[cid]['images'], frames):
                    target = out / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(io.BytesIO(png)).convert('RGB').save(target)
                for name in (public[cid]['board'], public[cid]['input_preview'], *public[cid]['images']):
                    assert sha256_path(out / name) == sha256_path(dataset / name), name
                checked.append(cid)
            print(json.dumps({'asset_id': aid, 'reproduced_cases': len(checked)}), flush=True)
    validation = {'status': 'passed', 'cases': len(checked), 'independent_objects': len(checked)//2,
        'native_image_files_reproduced': 6 * len(checked), 'native_image_size': [256, 256],
        'source_manifest_sha256': sha256_path(dataset / 'manifest.json'),
        'checker_replayed': True, 'all_image_sha256_matched': True, 'source_files_changed': 0}
    (out / 'verification.json').write_text(json.dumps(validation, indent=2)+'\n')
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.dataset.resolve(), args.out.resolve())))
