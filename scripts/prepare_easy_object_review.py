"""Train-only normal-first Easy Object screening; blind boards, private diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.prepare_obvious_defect_review import (
    DEFAULT_CONFIG, DEFAULT_SPLITS, ROOT, STATES, candidate_attempts, checker_result,
    image_metrics, render_program, sha256_path, write_jsonl,
)
from turnitover.assets.catalog import CatalogSource
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

DEFAULT_OUT = ROOT / 'docs/evidence/partnet-mobility-easy-objects-v2'


def case_id(asset, kind, seed):
    return 'eor-' + hashlib.sha256(f'easy-v2:{asset}:{kind}:{seed}'.encode()).hexdigest()[:12]


def normal_view(render):
    scores = {v: max(image_metrics(render['frames'][s][v], render['frames']['rest'][v])
                     ['core_changed_fraction'] for s in ('mid', 'limit'))
              for v in render['frames']['rest']}
    return max(scores, key=scores.get), scores


def gate(kind, metrics, checks):
    """Geometric/visibility screening only; never certifies human diagnosis."""
    reasons = []
    if not checks['checks'][1]['passed']:
        reasons.append('runtime_budget_failed')
    if checks['code_correct'] != (kind == 'clean'):
        reasons.append('independent_checker_disagreement')
    if kind == 'joint_axis':
        if metrics['rest']['changed_fraction'] > 0.001:
            reasons.append('axis_changed_rest')
        if max(metrics[s]['core_changed_fraction'] for s in ('mid', 'limit')) < .02:
            reasons.append('motion_visibility_insufficient')
    if kind == 'part_offset' and (metrics['rest']['core_changed_fraction'] < .01 or
                                 metrics['rest']['largest_core_region_px'] < 300):
        reasons.append('rest_misalignment_visibility_insufficient')
    return reasons


def board(path, asset, view, reference, candidates):
    tile, left, top, gap = 256, 145, 90, 10
    canvas = Image.new('RGB', (953, top + 4 * 266 + 10), '#fafaf8')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans.ttf', 14)
    draw.text((10, 10), f'{asset} | view {view} | native 256 x 256 | pending_human_review', fill='black', font=font)
    draw.text((10, 34), 'Compare each candidate with reference; review type, joint, part and evidence states.', fill='black', font=font)
    rows = [('Reference', reference)] + [(r['case_id'], r['render']) for r in candidates]
    for col, state in enumerate(STATES):
        x = left + col * 266
        draw.text((x, 64), f'{state} ({reference["values"][state]:.3f})', fill='black', font=font)
        for i, (label, rendered) in enumerate(rows):
            y = top + i * 266
            draw.text((5, y + 112), label, fill='black', font=font)
            canvas.paste(Image.open(io.BytesIO(rendered['frames'][state][view])).convert('RGB'), (x, y))
    path.parent.mkdir(exist_ok=True, parents=True)
    canvas.save(path)


def run(config, out):
    if out.exists():
        raise ValueError('Use a fresh output directory')
    cfg = load_generate_config(DEFAULT_CONFIG)
    assert (cfg.render.width, cfg.render.height) == (256, 256)
    cat = CatalogSource(cfg.asset_source['manifest'])
    splits = json.loads(DEFAULT_SPLITS.read_text())
    plan = json.loads(config.read_text())
    baseline = ROOT / 'docs/evidence/partnet-mobility-obvious-defects-v1'
    baseline_ids = {json.loads(x)['asset_id'] for x in (baseline/'review_manifest.jsonl').read_text().splitlines()}
    out.mkdir(parents=True)
    (out/'private').mkdir()
    views = load_views(cfg.views_path)
    public, gold, checks, screening, selections = [], [], [], [], []
    # Finish and persist all normal-only selections BEFORE creating any corruptions.
    with ObservationSession(cfg.render, views) as session:
        references = {}
        for row in plan['normal_shortlist']:
            aid = row['asset_id']
            assert aid not in baseline_ids
            assert splits['groups'][f'PartNet-Mobility:{aid[3:]}'] == 'train'
            spec = cat.get(aid)
            rendered = render_program(session, ObjectProgram.from_spec(spec), None, row['joint_id'], [str(v.id) for v in views])
            view, scores = normal_view(rendered)
            selection = {**row, 'view': view, 'normal_motion_scores': scores,
                         'difficulty': 'easy' if row['selected'] else 'excluded',
                         'selection_basis': 'normal asset only; AI operator prescreen; human confirmation required'}
            selections.append(selection)
            if row['selected']:
                from scripts.prepare_obvious_defect_review import state_values
                rendered['values'] = state_values(rendered['joint'])
                references[aid] = (spec, rendered, view)
            print('normal', aid, row['selected'], view, flush=True)
        assert len(references) == 8
        write_jsonl(out/'private/object_selection.jsonl', selections)
        selection_hash = sha256_path(out/'private/object_selection.jsonl')
        for index, (aid, (spec, ref, view)) in enumerate(references.items()):
            normal = ObjectProgram.from_spec(spec)
            target = next(r['joint_id'] for r in selections if r['asset_id'] == aid)
            variants = [{'kind': 'clean', 'seed': None, 'program': normal, 'render': ref, 'label': None}]
            for kind, base in [('joint_axis', 4100), ('part_offset', 5100)]:
                attempts, wrong = candidate_attempts(normal, target, kind, base + index * 100)
                screened = []
                for attempt in attempts:
                    rendered = render_program(session, attempt['program'], ref['framing'], target, [view])
                    metrics = {s: image_metrics(rendered['frames'][s][view], ref['frames'][s][view]) for s in STATES}
                    check = checker_result(attempt['program'], rendered, normal, ref)
                    reasons = gate(kind, metrics, check)
                    screened.append({**attempt, 'render': rendered, 'metrics': metrics, 'check': check, 'reasons': reasons})
                chosen = next((a for a in screened if not a['reasons']), screened[0])
                variants.append(chosen)
                screening.append({'asset_id': aid, 'kind': kind, 'wrong_target_attempts': wrong,
                                  'attempts': [{'seed': a['seed'], 'metrics': a['metrics'], 'checks': a['check'],
                                                'automatic_pass': not a['reasons'], 'rejection_reasons': a['reasons'],
                                                'selected': a is chosen} for a in screened]})
            for v in variants:
                v['case_id'] = case_id(aid, v['kind'], v['seed'])
                result = checker_result(v['program'], v['render'], normal, ref)
                metrics = {s: image_metrics(v['render']['frames'][s][view], ref['frames'][s][view]) for s in STATES}
                reasons = gate(v['kind'], metrics, result)
                status = 'rejected_auto_screen' if reasons else 'pending_human_review'
                public.append({'case_id': v['case_id'], 'asset_id': aid, 'board': f'boards/{aid}.png',
                               'view': view, 'states': list(STATES), 'status': status,
                               'hierarchy': ref['hierarchy'], 'joint_states': ref['values']})
                gold.append({'case_id': v['case_id'], 'kind': v['kind'], 'joint_id': target if v['kind'] != 'clean' else None,
                             'parts': [] if v['label'] is None else list(v['label'].parts),
                             'label': None if v['label'] is None else to_dict(v['label']),
                             'seed': v['seed'], 'candidate_sha': v['program'].sha, 'reference_sha': normal.sha,
                             'training_eligibility': status, 'rejection_reasons': reasons})
                checks.append({'case_id': v['case_id'], **result})
                # Private full specs allow exact regeneration without relying on seed alone.
                (out/'private'/f'{v["case_id"]}.json').write_text(json.dumps(to_dict(v['program'].spec)))
            board(out/f'boards/{aid}.png', aid, view, ref, sorted(variants, key=lambda v: v['case_id']))
            print('candidates', aid, flush=True)
    write_jsonl(out/'review_manifest.jsonl', sorted(public, key=lambda r:r['case_id']))
    write_jsonl(out/'human_review.jsonl', [{'case_id': r['case_id'], 'status': r['status'],
        'human_judgment': {'verdict': None, 'defect_id': None, 'joint_id': None, 'parts': [], 'evidence_states': [], 'notes': None}} for r in sorted(public, key=lambda r:r['case_id'])])
    write_jsonl(out/'private/gold.jsonl', gold)
    write_jsonl(out/'private/checks.jsonl', checks)
    write_jsonl(out/'private/screening.jsonl', screening)
    summary = {'easy_objects': {'shortlisted': len(selections), 'passed': len(references),
                               'train_pool': sum(v=='train' for v in splits['groups'].values()),
                               'baseline_assets_excluded': sorted(baseline_ids)},
               'types': {k: {'attempts': sum(len(r['attempts']) for r in screening if r['kind']==k),
                            'automatic_pass': sum(a['automatic_pass'] for r in screening if r['kind']==k for a in r['attempts']),
                            'selected_pending': sum(g['kind']==k and g['training_eligibility']=='pending_human_review' for g in gold)} for k in ('joint_axis','part_offset')},
               'cases': len(public), 'boards': len(references),
               'pending_human_review': sum(r['status']=='pending_human_review' for r in public),
               'model_calls': 0, 'sft_training_runs': 0, 'test_rows_touched': 0,
               'selection_sha256_before_injection': selection_hash,
               'claim_scope': 'Automatic geometry and visibility only; human semantic diagnosability unconfirmed.'}
    (out/'private/summary.json').write_text(json.dumps(summary, indent=2))
    inputs = [config, DEFAULT_CONFIG, DEFAULT_SPLITS, Path(cfg.asset_source['manifest']), Path(cfg.views_path), Path(__file__), ROOT/'scripts/prepare_obvious_defect_review.py']
    manifest = {'inputs': {str(p): sha256_path(p) for p in inputs},
                'baseline_hashes': {str(p.relative_to(ROOT)): sha256_path(p) for p in baseline.rglob('*') if p.is_file()},
                'files': {str(p.relative_to(out)): sha256_path(p) for p in out.rglob('*') if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/easy_object_review_v2.json')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.config, args.out)
