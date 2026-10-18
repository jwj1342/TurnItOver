"""Expand Level 0 using only the existing Train catalog and unchanged v1 gates."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from scripts.prepare_level0_axis_review import (
    AXIS_DEG, DEFAULT_OUT as V1, axis_attempts, board, full_normal_framing,
    normal_view, opaque_id, pair_preview, score, screen,
)
from scripts.prepare_obvious_defect_review import (
    DEFAULT_CONFIG, DEFAULT_SPLITS, ROOT, STATES, checker_result, render_program,
    sha256_path, state_values, write_jsonl,
)
from turnitover.assets.catalog import CatalogSource
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

DEFAULT_OUT = ROOT / 'docs/evidence/partnet-mobility-level0-axis-v2'


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seal(out: Path, inputs: dict) -> None:
    manifest = {'version': 2, 'scope': 'Train only; Level 0 joint_axis', 'inputs': inputs,
                'files': {p.relative_to(out).as_posix(): sha256_path(p)
                          for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def find_pair(session, reference, views, seed_base):
    """Try each joint; view ranking and framing depend only on normal motion."""
    diagnostics = []
    for joint_index, joint in enumerate(reference.spec.joints):
        initial = render_program(session, reference, None, joint.id, views)
        framing = full_normal_framing(initial)
        ref = render_program(session, reference, framing, joint.id, views)
        _, scores = normal_view(ref)
        clean_check = checker_result(reference, ref, reference, ref)
        joint_log = {'joint_id': joint.id, 'normal_framing': to_dict(framing),
                     'state_values': state_values(ref['joint']),
                     'normal_view_scores': scores, 'attempts': []}
        diagnostics.append(joint_log)
        if not clean_check['code_correct']:
            joint_log['rejection_reasons'] = ['clean_checker_failed']
            continue
        # Try all normal-motion-eligible views without changing framing or thresholds.
        eligible_views = [v for v in sorted(scores, key=lambda v: (-scores[v], v)) if scores[v] >= .020]
        if not eligible_views:
            joint_log['rejection_reasons'] = ['reference_motion_too_small_all_views']
            continue
        best = None
        for attempt in axis_attempts(reference, joint.id, seed_base + joint_index * 200):
            rendered = render_program(session, attempt['program'], framing, joint.id, eligible_views)
            check = checker_result(attempt['program'], rendered, reference, ref)
            for view in eligible_views:
                metrics, reasons = screen(ref, rendered, view, check)
                joint_log['attempts'].append({'seed': attempt['seed'], 'view': view,
                    'automatic_pass': not reasons, 'rejection_reasons': reasons, 'metrics': metrics})
                if not reasons and (best is None or score(metrics) > score(best['metrics'])):
                    best = {**attempt, 'render': rendered, 'check': check, 'view': view, 'metrics': metrics}
        if best:
            joint_log['selected'] = {'seed': best['seed'], 'view': best['view']}
            return (ref, best, joint.id), diagnostics
    return None, diagnostics


def run(config_path=DEFAULT_CONFIG, splits_path=DEFAULT_SPLITS, out=DEFAULT_OUT, source=V1):
    if out.exists():
        raise ValueError(f'Output must not exist: {out}')
    cfg = load_generate_config(config_path)
    if (cfg.render.width, cfg.render.height) != (256, 256):
        raise ValueError('Level 0 requires native 256x256 renders')
    splits = json.loads(splits_path.read_text())
    catalog = CatalogSource(cfg.asset_source['manifest'])
    train = [s for s in catalog if splits['groups'][catalog.metadata(s.asset_id)['split_group']] == 'train']
    source_manifest = json.loads((source / 'manifest.json').read_text())
    for p, digest in source_manifest['files'].items():
        if sha256_path(source / p) != digest:
            raise ValueError(f'v1 SHA mismatch: {p}')
    public = read_rows(source / 'review_manifest.jsonl')
    gold = read_rows(source / 'private/gold.jsonl')
    checks = read_rows(source / 'private/checks.jsonl')
    reviews = read_rows(source / 'human_review.jsonl')
    inherited = {g['asset_id'] for g in gold}
    if not inherited <= {s.asset_id for s in train}:
        raise ValueError('Inherited objects must be Train')
    out.mkdir(parents=True)
    (out / 'private').mkdir()
    for row in public:
        for key in ('board', 'input_preview'):
            target = out / row[key]
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source / row[key], target)
    for g in gold:
        shutil.copyfile(source / 'private' / f"{g['case_id']}.json", out / 'private' / f"{g['case_id']}.json")
    # Preserve actual prior judgments verbatim; provenance is separate from the judgment.
    provenance = {r['case_id']: {'source': str(source.relative_to(ROOT)),
        'review_file_sha256': sha256_path(source / 'human_review.jsonl'),
        'source_manifest_sha256': sha256_path(source / 'manifest.json')} for r in reviews}
    (out / 'private/inherited_review_provenance.json').write_text(json.dumps(provenance, indent=2))
    shutil.copyfile(source / 'private/screening.json', out / 'private/inherited_screening.json')
    audit = []

    def checkpoint():
        write_jsonl(out / 'review_manifest.jsonl', sorted(public, key=lambda r: r['case_id']))
        write_jsonl(out / 'private/gold.jsonl', gold)
        write_jsonl(out / 'private/checks.jsonl', checks)
        write_jsonl(out / 'human_review.jsonl', sorted(reviews, key=lambda r: r['case_id']))
        (out / 'private/screening.json').write_text(json.dumps({'assets': audit}, indent=2))

    checkpoint()
    views = load_views(cfg.views_path)
    with ObservationSession(cfg.render, views) as session:
        for index, spec in enumerate(train):
            if spec.asset_id in inherited:
                audit.append({'asset_id': spec.asset_id, 'status': 'inherited_v1'})
                continue
            reference = ObjectProgram.from_spec(spec)
            pair, diagnostics = find_pair(session, reference, [str(v.id) for v in views], 20000 + index * 1000)
            counts = Counter(reason for j in diagnostics for a in j['attempts'] for reason in a['rejection_reasons'])
            for j in diagnostics:
                counts.update(j.get('rejection_reasons', []))
            audit.append({'asset_id': spec.asset_id, 'status': 'automatic_pass' if pair else 'rejected',
                          'rejection_reason_counts': dict(counts), 'joints': diagnostics})
            print(json.dumps({'asset': spec.asset_id, 'automatic_pass': pair is not None, 'reasons': dict(counts)}), flush=True)
            if pair:
                ref, chosen, joint_id = pair
                view = chosen['view']
                for kind, seed, program, rendered, label in (
                    ('clean', None, reference, ref, None),
                    ('joint_axis', chosen['seed'], chosen['program'], chosen['render'], chosen['label'])):
                    case_id = opaque_id(spec.asset_id, kind, seed)
                    bp, ip = Path('boards') / f'{case_id}.png', Path('inputs') / f'{case_id}.png'
                    board(out / bp, case_id, ref, rendered, view)
                    pair_preview(out / ip, ref, rendered, view)
                    public.append({'case_id': case_id, 'category': spec.category, 'view': view,
                        'states': list(STATES), 'input_states': ['rest', 'limit'], 'image_size': [256, 256],
                        'board': bp.as_posix(), 'input_preview': ip.as_posix(), 'status': 'pending_human_review'})
                    gold.append({'case_id': case_id, 'asset_id': spec.asset_id, 'kind': kind,
                        'expected_status': 'pass' if kind == 'clean' else 'fail',
                        'defect_id': None if label is None else label.defect_id,
                        'joint_id': None if label is None else joint_id,
                        'parts': [] if label is None else list(label.parts),
                        'parameters': {} if label is None else label.params,
                        'candidate_sha': program.sha, 'reference_sha': reference.sha,
                        'training_eligibility': 'pending_human_review'})
                    checks.append({'case_id': case_id, **checker_result(program, rendered, reference, ref)})
                    reviews.append({'case_id': case_id, 'status': 'pending_human_review',
                        'reviewer': None, 'reviewed_at': None,
                        'human_judgment': {'verdict': None, 'defect_id': None, 'one_glance': None,
                            'normal_vs_candidate_motion': None, 'evidence_states': [], 'notes': None,
                            'no_clipping_or_occlusion': None, 'rest_matches_reference': None}})
                    (out / 'private' / f'{case_id}.json').write_text(json.dumps(to_dict(program.spec)))
            checkpoint()
    inputs = {str(p.resolve()): sha256_path(p) for p in
        (config_path, splits_path, Path(cfg.asset_source['manifest']), Path(cfg.views_path),
         Path(__file__), ROOT / 'scripts/prepare_level0_axis_review.py',
         ROOT / 'scripts/prepare_obvious_defect_review.py', source / 'manifest.json')}
    seal(out, inputs)
    from scripts.finalize_level0_axis_v2 import finalize
    return finalize(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--splits', type=Path, default=DEFAULT_SPLITS)
    parser.add_argument('--source', type=Path, default=V1)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.splits.resolve(), args.out.resolve(), args.source.resolve())))
