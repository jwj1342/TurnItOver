"""Prepare private truth and frozen Chromium witnesses for the reviewed Easy cohort."""
from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_obvious_defect_review import (
    DEFAULT_CONFIG, DEFAULT_SPLITS, ROOT, checker_result, image_metrics,
    render_program, sha256_path, write_jsonl,
)
from turnitover.assets.catalog import CatalogSource, decode_spec
from turnitover.config import load_generate_config
from turnitover.core.actions import RuntimeProperty
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views
from turnitover.teacher.witnesses import row_sha256, rest_state
from turnitover.verifier.contracts import Context

BUNDLE = ROOT / 'docs/evidence/partnet-mobility-easy-objects-v2'
COHORT = ROOT / 'docs/evidence/partnet-mobility-easy-objects-v2-audit/high_confidence_candidates.jsonl'
PROTOCOL = 'easy_selected_joint_motion_v1'
PILOT = 'eor-57a7ca1300ca'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cohort():
    rows = read_rows(COHORT)
    assert len(rows) == 18 and len({r['asset_id'] for r in rows}) == 7
    assert {k: sum(r['kind'] == k for r in rows) for k in ('clean', 'joint_axis', 'part_offset')} == {
        'clean': 7, 'joint_axis': 6, 'part_offset': 5}
    assert all(r['high_confidence_candidate'] for r in rows)
    return sorted(rows, key=lambda r: (r['case_id'] != PILOT, r['case_id']))


def task_text(spec, joint_id, view):
    joint = spec.joint(joint_id)
    return (f'Object category: {spec.category}. Scope=easy_selected_joint_motion_v1. '
            f'Compare the candidate with the normal reference for joint {joint_id} '
            f'({joint.type.value if hasattr(joint.type, "value") else joint.type}), '
            f'moving part {joint.part}, at view {view}, rest/mid/limit. '
            'Other joints remain at rest and are outside this declared motion scope. '
            'The initial camera is the reference view. Joint and camera state persist. '
            'Choose your own observations and when to finish. A scope-complete pass needs '
            'candidate observations covering every distinct supplied reference state; '
            'a fail must cite only observations visibly supporting its type and location. '
            'Use the full taxonomy; do not assume the candidate contains a defect. '
            'Only reference images, public hierarchy, runtime queries and executed images are visible.')


def _board_pixels(case, rendered, view):
    board = Image.open(BUNDLE / 'boards' / f'{case["asset_id"]}.png').convert('RGB')
    row = case['board_row'] - 1
    for col, state in enumerate(('rest', 'mid', 'limit')):
        tile = board.crop((145 + col * 266, 90 + row * 266,
                           145 + col * 266 + 256, 90 + row * 266 + 256))
        live = Image.open(io.BytesIO(rendered['frames'][state][view])).convert('RGB')
        if not np.array_equal(np.asarray(tile), np.asarray(live)):
            raise ValueError(f'Fresh Chromium differs from human-reviewed board: {case["case_id"]}/{state}')


def prepare(out: Path):
    if out.exists():
        raise ValueError('Preparation requires a new output directory')
    out.mkdir(parents=True)
    private = out / 'private'
    private.mkdir()
    cfg = load_generate_config(DEFAULT_CONFIG)
    assert (cfg.render.width, cfg.render.height) == (256, 256)
    catalog = CatalogSource(cfg.asset_source['manifest'])
    splits = json.loads(DEFAULT_SPLITS.read_text())['groups']
    views = load_views(cfg.views_path)
    gold_by_id = {r['case_id']: r for r in read_rows(BUNDLE / 'private/gold.jsonl')}
    public = {r['case_id']: r for r in read_rows(BUNDLE / 'review_manifest.jsonl')}
    selected = {r['asset_id']: r for r in read_rows(BUNDLE / 'private/object_selection.jsonl') if r['selected']}
    refs, prepared, annotations, golds = {}, [], [], []
    with ObservationSession(cfg.render, views) as session:
        browser = session.browser_version()
        for case in cohort():
            cid, aid = case['case_id'], case['asset_id']
            metadata = catalog.metadata(aid)
            assert splits['PartNet-Mobility:' + aid[3:]] == 'train'
            spec = catalog.get(aid)
            joint, view = selected[aid]['joint_id'], selected[aid]['view']
            normal = ObjectProgram.from_spec(spec)
            if aid not in refs:
                refs[aid] = render_program(session, normal, None, joint, [view])
            ref = refs[aid]
            program = ObjectProgram.from_spec(decode_spec(json.loads(
                (BUNDLE / f'private/{cid}.json').read_text())))
            expected = gold_by_id[cid]
            assert program.sha == expected['candidate_sha'] and normal.sha == expected['reference_sha']
            rendered = render_program(session, program, ref['framing'], joint, [view])
            _board_pixels(case, rendered, view)
            check = checker_result(program, rendered, normal, ref)
            assert check['code_correct'] == (case['kind'] == 'clean')
            folder = private / 'frozen' / cid
            folder.mkdir(parents=True)
            info = session.load(program, framing=ref['framing'])
            hierarchy = session.query_runtime(RuntimeProperty.HIERARCHY)
            session.dispose()
            limits = {k: list(v['limits']) for k, v in info.joints.items()}
            states, witnesses, signals = [], [], []
            clean = case['kind'] == 'clean'
            findings = [] if clean else [{
                'defect_id': 'kinematics.joint_axis' if case['kind'] == 'joint_axis' else 'structure.part_offset',
                'parts': expected['parts'], 'joint_id': joint if case['kind'] == 'joint_axis' else None,
                'description': f'Private checker-qualified defect record {cid}.'}]
            identity = {**metadata, 'source_id': catalog.source_id, 'asset_id': aid,
                        'split': 'train', 'split_group': 'PartNet-Mobility:' + aid[3:],
                        'candidate_sha': program.sha, 'reference_sha': normal.sha,
                        'experiment_mode': 'active', 'reference_protocol': PROTOCOL,
                        'common_input': 'hierarchy', 'prompt_addenda': ['part_convention.txt']}
            gold = {'id': cid, 'asset_id': aid, 'split_group': identity['split_group'],
                    'source_id': catalog.source_id, 'program_sha': program.sha,
                    'reference_sha': normal.sha, 'provenance': metadata,
                    'quality': {'status': 'accepted'},
                    'target': {'status': 'pass' if clean else 'fail', 'findings': findings}}
            annotation = {'id': cid, 'asset_id': aid, 'split': 'train',
                          'split_group': identity['split_group'], 'candidate_sha': program.sha,
                          'reference_sha': normal.sha, 'data_identity': identity,
                          'labels': findings, 'signals': {'against_reference': signals}}
            for index, state_name in enumerate(('rest', 'mid', 'limit')):
                state = rest_state(limits)
                if state_name != 'rest':
                    state[joint] = sum(limits[joint]) / 2 if state_name == 'mid' else limits[joint][1]
                label = {'view': view, 'joint_id': joint, 'detent': state_name}
                path = folder / f'reference-{index:02d}.png'
                path.write_bytes(ref['frames'][state_name][view])
                states.append({'file': path.name, 'label': label, 'state': state, 'sha256': sha256_path(path)})
                candidate_path = folder / f'candidate-{index:02d}.png'
                candidate_path.write_bytes(rendered['frames'][state_name][view])
                metric = image_metrics(rendered['frames'][state_name][view], ref['frames'][state_name][view])
                geometry_state = 'rest' if state_name == 'rest' else f'joint:target:{state_name}'
                geom_failed = any(not inv['passed'] and inv['state'] == geometry_state
                                  and set(inv['parts']) & set(expected['parts'])
                                  for inv in check['checks'][0]['invariants'])
                exposed = not clean and geom_failed and metric['core_changed_fraction'] >= .001
                signals.append(metric)
                witnesses.append({'index': index, 'key': f'{view}:{joint}:{state_name}',
                                  'view': view, 'state': state, 'sha256': sha256_path(candidate_path),
                                  'exposures': [0] if exposed else [],
                                  'perceptible_exposures': [0] if exposed else []})
            manifest = {'schema_version': 2, 'id': cid, 'asset_id': aid, 'split': 'train',
                        'split_group': identity['split_group'], 'candidate_sha': program.sha,
                        'reference_sha': normal.sha, 'in_frozen_test': False,
                        'split_verification': 'Train source split identity; frozen Test never opened',
                        'candidate_joint_limits': limits, 'reference_joint_limits': limits,
                        'row_sha256': {'gold': row_sha256(gold), 'annotation': row_sha256(annotation)},
                        'framing': to_dict(ref['framing']), 'witnesses': witnesses,
                        'rest_hierarchy': {'candidate': hierarchy},
                        'reference_protocol': {'name': PROTOCOL, 'states': states, 'label_independent': True},
                        'perceptibility': {'min_core_fraction': .001, 'geometry_failure_required': True},
                        'checker': check, 'reviewed_board_pixel_match': True}
            (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            context = Context(task_text(spec, joint, view), spec.part_ids, (joint,), (view,),
                              ('hierarchy', 'joint_state', 'stats'),
                              ('request_view', 'actuate_joint', 'query_runtime'), 1_000_000, 1024)
            prepared.append({'case_id': cid, 'identity': identity, 'context': asdict(context),
                             'manifest': str((folder / 'manifest.json').relative_to(out))})
            golds.append(gold)
            annotations.append(annotation)
            print(json.dumps({'phase': 'prepared', 'case_id': cid}), flush=True)
    write_jsonl(private / 'gold.jsonl', golds)
    write_jsonl(private / 'annotations.jsonl', annotations)
    write_jsonl(out / 'prepared.jsonl', prepared)
    return {'browser_version': browser, 'cases': len(prepared), 'reviewed_board_pixel_matches': len(prepared),
            'cohort_sha256': sha256_path(COHORT)}
