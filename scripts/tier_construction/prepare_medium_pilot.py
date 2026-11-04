"""Construct a resumable, human-pending Medium pilot: 5 pairs in each of 3 configurations."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import random
import tempfile

from scripts.tier_construction.construction_pilot import write_frames
from scripts.tier_construction.medium_pilot_search import CONFIGS, find_pair
from scripts.tier_construction.prepare_level0_axis_review import board, pair_preview
from scripts.tier_construction.prepare_obvious_defect_review import DEFAULT_CONFIG, ROOT, STATES, checker_result, sha256_path
from turnitover.assets.audit_urdf import audit_model
from turnitover.assets.urdf import UrdfModel
from turnitover.config import load_generate_config
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.render.session import ObservationSession
from turnitover.render.views import load_views

OUT = ROOT / 'benchmark/medium-pilot-v1'
PER_CONFIG = 5
MAX_TRIES = 80
POLICY = 'medium-pilot-v1; fixed normal camera; first passing seed; no human approval inferred'


def _id(*parts):
    return 'mp-' + hashlib.sha256(':'.join(map(str, (POLICY, *parts))).encode()).hexdigest()[:14]


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in values))
    temporary.replace(path)


def _review(cid):
    return {'case_id': cid, 'status': 'pending_human_review', 'reviewer': None, 'reviewed_at': None,
            'human_judgment': {'verdict': None, 'defect_id': None, 'one_glance': None,
                'medium_obviousness': None, 'what_is_wrong': None, 'flagged_parts': None,
                'flagged_part_is_the_wrong_one': None, 'rest_matches_reference': None,
                'rest_difference_visible': None, 'mid_difference_visible': None, 'limit_difference_visible': None,
                'evidence_states': [], 'no_clipping_or_occlusion': None, 'notes': None}}


def _page(out, public, name='review.html'):
    cards = '\n'.join(f'<section><h3>{html.escape(r["case_id"])}</h3>'
        f'<img src="{html.escape(r["input_preview"])}" alt="four native tiles">'
        f'<img src="{html.escape(r["board"])}" alt="rest mid limit comparison"></section>' for r in public)
    (out / name).write_text('<!doctype html><meta charset="utf-8"><title>Blind visual review</title>'
        '<style>body{font-family:sans-serif;max-width:960px;margin:2em auto}img{max-width:100%}'
        'section{border-bottom:1px solid #ddd;padding:2em 0}</style><h1>盲审</h1>'
        '<p>四图顺序：参考 rest、参考 limit、候选 rest、候选 limit。图板上排参考、下排候选，'
        '列为 rest / mid / limit。同一固定相机。逐例填写 human_review.jsonl，先独立观察图像。'
        '请判断是否正确、错误部件及位置或运动变化、一眼明显性、各状态证据、出框及遮挡。</p>' + cards)


def _load_records(out):
    records = []
    sealed = out / 'manifest.json'
    if sealed.exists():
        for name, digest in json.loads(sealed.read_text())['files'].items():
            if name.startswith(('private/objects/', 'private/rejections/')) and sha256_path(out / name) != digest:
                raise ValueError(f'Completed object record SHA mismatch: {name}')
    paths = list((out / 'private/objects').glob('*/record.json')) + list((out / 'private/rejections').glob('*.json'))
    for path in sorted(paths):
        record = json.loads(path.read_text())
        for name, digest in record['files'].items():
            if sha256_path(out / name) != digest:
                raise ValueError(f'Completed object artifact SHA mismatch: {name}')
        if record['status'] == 'automatic_pass':
            if any(len(record.get(key, [])) != 2 for key in ('public', 'gold', 'checks', 'reviews')):
                raise ValueError('Incomplete accepted object checkpoint')
            if record['source_authenticity'].get('passed') is not True:
                raise ValueError('Accepted object lacks source authenticity')
        records.append(record)
    keys = [(r['config'], r['asset_id']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate object checkpoint')
    accepted = [r for r in records if r['status'] == 'automatic_pass']
    if len({r['asset_id'] for r in accepted}) != len(accepted):
        raise ValueError('Accepted objects must be independent across configurations')
    return records


def seal(out, inputs):
    for name, digest in inputs.items():
        if sha256_path(Path(name)) != digest:
            raise ValueError(f'Generation input SHA changed: {name}')
    manifest = {'version': 1, 'scope': 'Medium pilot only; Train candidates; all human pending',
                'inputs': inputs, 'files': {p.relative_to(out).as_posix(): sha256_path(p)
                    for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'},
                'policy': POLICY, 'per_config_hard_limit': PER_CONFIG}
    _json(out / 'manifest.json', manifest)


def checkpoint(out, records, inputs):
    public = sorted((p for r in records for p in r.get('public', [])), key=lambda p: p['case_id'])
    gold = sorted((g for r in records for g in r.get('gold', [])), key=lambda g: g['case_id'])
    reviews = sorted((v for r in records for v in r.get('reviews', [])), key=lambda v: v['case_id'])
    _jsonl(out / 'review_manifest.jsonl', public)
    _jsonl(out / 'private/gold.jsonl', gold)
    _jsonl(out / 'private/checks.jsonl', sorted((c for r in records for c in r.get('checks', [])), key=lambda c: c['case_id']))
    _jsonl(out / 'human_review.jsonl', reviews)
    _jsonl(out / 'train_manifest.jsonl', [])
    _jsonl(out / 'private/train_targets.jsonl', [])
    _json(out / 'private/screening.json', {'assets': [{k: v for k, v in r.items()
        if k not in {'public', 'gold', 'checks', 'reviews', 'files'}} for r in records]})
    _page(out, public)
    sampled = random.Random(20261023).sample(public, min(5, len(public)))
    _json(out / 'spot_check.json', {'seed': 20261023, 'case_ids': [p['case_id'] for p in sampled],
                                  'role': 'additional random inspection; does not replace all-case human review'})
    _page(out, sampled, 'review-spot.html')
    counts = Counter(r['config'] for r in records if r['status'] == 'automatic_pass')
    report = {'status': 'automatic_candidates_ready' if all(counts[c] == PER_CONFIG for c in CONFIGS)
              else 'pilot_incomplete', 'human_review_status': 'pending_human_review',
              'independent_objects': sum(counts.values()), 'cases': len(public),
              'configs': {c: {'target': PER_CONFIG, 'objects': counts[c],
                  'tried': sum(r['config'] == c for r in records)} for c in CONFIGS},
              'human_review_complete': 0, 'training_eligible_objects': 0, 'training_eligible_cases': 0,
              'native_image_size': [256, 256], 'model_calls': 0, 'training_runs': 0, 'paid_api_calls': 0}
    _json(out / 'validation.json', report)
    seal(out, inputs)
    return report


def _artifacts(stage, item, pair, configuration, reference):
    ref, chosen, joint_id, values = pair
    source = {'split': 'train', **item['metadata'], 'source_urdf': item['source_urdf'],
              'complexity': item['complexity']}
    rows = {'public': [], 'gold': [], 'checks': [], 'reviews': []}
    for kind, program, rendered, label, seed in (
            ('clean', reference, ref, None, None),
            (CONFIGS[configuration]['defect'].split('.')[1], chosen['program'], chosen['render'], chosen['label'], chosen['seed'])):
        cid = _id(configuration, reference.spec.asset_id, kind, seed)
        view = chosen['view']
        result = checker_result(program, rendered, reference, ref)
        if result['code_correct'] != (label is None) or not result['checks'][1]['passed']:
            raise ValueError(f'Checker disagreement while exporting {cid}')
        board(stage / 'boards' / f'{cid}.png', cid, ref, rendered, view)
        pair_preview(stage / 'inputs' / f'{cid}.png', ref, rendered, view)
        write_frames(stage, cid, ref, rendered, view)
        for name, data in (('reference', ref), ('candidate', rendered)):
            for state in STATES:
                path = stage / 'native' / cid / f'{name}_{state}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data['frames'][state][view])
        program_path = f'private/programs/{cid}.json'
        _json(stage / program_path, to_dict(program.spec))
        rows['public'].append({'case_id': cid, 'category': reference.spec.category, 'view': view,
            'states': list(STATES), 'input_states': ['rest', 'limit'], 'image_size': [256, 256],
            'board': f'boards/{cid}.png', 'input_preview': f'inputs/{cid}.png',
            'images': [f'frames/{cid}/{i}.png' for i in range(4)], 'status': 'pending_human_review'})
        rows['gold'].append({'case_id': cid, 'asset_id': reference.spec.asset_id, 'config': configuration,
            'difficulty': 'medium_provisional', 'kind': kind, 'expected_status': 'pass' if label is None else 'fail',
            'defect_id': None if label is None else label.defect_id, 'joint_id': joint_id,
            'parts': [] if label is None else list(label.parts), 'parameters': {} if label is None else label.params,
            'seed': seed, 'state_values': values, 'candidate_sha': program.sha, 'reference_sha': reference.sha,
            'program': program_path, 'source': source, 'training_eligibility': 'pending_human_review'})
        rows['checks'].append({'case_id': cid, **result})
        rows['reviews'].append(_review(cid))
    return rows


def run(out=OUT, per_config=PER_CONFIG, max_tries=MAX_TRIES):
    if not 1 <= per_config <= PER_CONFIG or not 1 <= max_tries <= MAX_TRIES:
        raise ValueError('This CLI is pilot-only: at most 5 objects per configuration and 1..80 attempts')
    out = Path(out).resolve()
    cfg = load_generate_config(DEFAULT_CONFIG)
    if (cfg.render.width, cfg.render.height) != (256, 256):
        raise ValueError('Medium pilot requires native 256x256 rendering')
    from scripts.tier_construction.medium_pilot_sources import load_sources
    pool, provenance = load_sources(out=out)
    views = load_views(cfg.views_path)
    inputs = dict(provenance)
    for path in (Path(__file__), ROOT / 'scripts/medium_pilot_search.py', ROOT / 'scripts/medium_pilot_sources.py',
                 ROOT / 'scripts/construction_pilot.py', ROOT / 'scripts/level0_expansion_search.py',
                 ROOT / 'scripts/validate_medium_pilot.py',
                 ROOT / 'scripts/prepare_level0_axis_review.py', ROOT / 'scripts/prepare_obvious_defect_review.py',
                 ROOT / 'turnitover/assets/audit_urdf.py', ROOT / 'turnitover/assets/urdf.py',
                 ROOT / 'turnitover/corruptions/kinematics.py', ROOT / 'turnitover/corruptions/structure.py', Path(DEFAULT_CONFIG), Path(cfg.views_path)):
        inputs[str(path.resolve())] = sha256_path(path)
    state_path = out / 'private/run_state.json'
    state = {'inputs': inputs, 'policy': POLICY, 'per_config_hard_limit': PER_CONFIG, 'max_tries': max_tries}
    if state_path.exists():
        if json.loads(state_path.read_text()) != state:
            raise ValueError('Resume code, configuration or source queue changed')
        for review in _rows(out / 'human_review.jsonl'):
            if review != _review(review['case_id']):
                raise ValueError('Generation cannot overwrite a modified human review')
    else:
        _json(state_path, state)
    records = _load_records(out)
    done = {(r['config'], r['asset_id']) for r in records}
    selected = {r['asset_id'] for r in records if r['status'] == 'automatic_pass'}
    checkpoint(out, records, inputs)
    with ObservationSession(cfg.render, views) as session:
        for configuration in CONFIGS:
            candidates = [r for r in pool if r['config'] == configuration]
            for item in candidates:
                tried = sum(r['config'] == configuration for r in records)
                accepted = sum(r['config'] == configuration and r['status'] == 'automatic_pass' for r in records)
                if accepted >= per_config or tried >= max_tries:
                    break
                spec = item['spec']
                if spec.asset_id in selected or (configuration, spec.asset_id) in done:
                    continue
                reference = ObjectProgram.from_spec(spec)
                seed = 7_000_000 + int(hashlib.sha256(f'{configuration}:{spec.asset_id}'.encode()).hexdigest()[:8], 16)
                pair, diagnostics = find_pair(session, reference, views, configuration, seed)
                authenticity = None
                if pair:
                    try:
                        authenticity = audit_model(session, UrdfModel(Path(item['source_urdf'])), spec)
                    except Exception as error:
                        authenticity = {'passed': False, 'error_type': type(error).__name__, 'error': str(error)}
                    if authenticity.get('passed') is not True:
                        pair = None
                record = {'asset_id': spec.asset_id, 'category': spec.category, 'config': configuration,
                    'status': 'automatic_pass' if pair else 'rejected', 'diagnostics': diagnostics,
                    'source_authenticity': authenticity, 'source': item['metadata'], 'files': {}}
                if pair:
                    _, chosen, _, values = pair
                    record['camera'] = {'view': chosen['view'], 'framing': to_dict(chosen['framing']),
                        'state_values': values, 'policy': 'normal-only frozen before corruption; no error-guided view retry'}
                    # Commit marker is written last. Interrupted objects never count
                    # as complete; their deterministic artifacts are regenerated.
                    with tempfile.TemporaryDirectory(prefix='medium-pilot-', dir=out.parent) as temporary:
                        stage = Path(temporary)
                        record.update(_artifacts(stage, item, pair, configuration, reference))
                        for path in sorted(stage.rglob('*')):
                            if path.is_file():
                                relative = path.relative_to(stage)
                                target = out / relative
                                target.parent.mkdir(parents=True, exist_ok=True)
                                path.replace(target)
                                record['files'][relative.as_posix()] = sha256_path(target)
                    selected.add(spec.asset_id)
                record_path = (out / 'private/objects' / _id(configuration, spec.asset_id) / 'record.json'
                               if pair else out / 'private/rejections' / f'{_id(configuration, spec.asset_id)}.json')
                _json(record_path, record)
                records.append(record)
                done.add((configuration, spec.asset_id))
                result = checkpoint(out, records, inputs)
                print(json.dumps({'config': configuration, 'asset_id': spec.asset_id,
                                  'status': record['status'], 'cases': result['cases']}), flush=True)
    return checkpoint(out, records, inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--per-config', type=int, default=PER_CONFIG)
    parser.add_argument('--max-tries', type=int, default=MAX_TRIES)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.per_config, args.max_tries)))
