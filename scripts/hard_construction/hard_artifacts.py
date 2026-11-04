"""Native artifacts and answer-free, two-view human-review pages for Hard pilots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from scripts.tier_construction.prepare_obvious_defect_review import checker_result
from scripts.hard_construction.hard_policy import POLICY
from turnitover.core.serde import to_dict

STATES = ('rest', 'mid', 'limit')
ASSETS = Path(__file__).with_name('hard_review_assets')


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _review(cid):
    return {'case_id': cid, 'status': 'pending_human_review', 'reviewer': None,
            'reviewed_at': None, 'human_judgment': {
                'initial_verdict': None, 'initial_notes': None, 'final_decision': None,
                'final_notes': None, 'supplementary_decisive': None,
                'wrong_part_description': None, 'observed_views': [], 'view_switches': []}}


def artifacts(out, item, pair, reference):
    """Copy actual rendered bytes; private truth never appears in public cards."""
    out = Path(out)
    ref, chosen, joint_id, values = pair
    initial, supplementary = chosen['initial_view'], chosen['supplementary_view']
    if initial == supplementary or any(not isinstance(v, str) or
            not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', v) for v in (initial, supplementary)):
        raise ValueError('Hard review requires two distinct normal-frozen views')
    clean_check = chosen.get('clean_check')
    if clean_check is None:
        clean_check = checker_result(reference, ref, reference, ref)
    if not clean_check['code_correct'] or chosen['check']['code_correct']:
        raise ValueError('Clean/defect checker disagreement at artifact export')
    if not all(c['checks'][1]['passed'] for c in (clean_check, chosen['check'])):
        raise ValueError('Runtime budget failed at artifact export')
    for program, check, clean in ((reference, clean_check, True),
                                  (chosen['program'], chosen['check'], False)):
        if (check['candidate_sha'] != program.sha or check['reference_sha'] != reference.sha
                or check['checks'][0]['passed'] is not clean):
            raise ValueError('Exact checker is not bound to the exported program and reference')
    source = {**item['metadata'], 'source_urdf': item['source_urdf'],
              'complexity': item['complexity'], 'split': 'train'}
    rows = {key: [] for key in ('public', 'gold', 'checks', 'reviews')}
    for kind, program, rendered, label, seed, check in (
        ('clean', reference, ref, None, None, clean_check),
        (chosen['label'].defect_id.rsplit('.', 1)[1], chosen['program'], chosen['render'],
         chosen['label'], chosen['seed'], chosen['check']),
    ):
        raw = f"{POLICY['version']}:{item['config']}:{reference.spec.asset_id}:{kind}:{seed}"
        cid = 'hp-' + hashlib.sha256(raw.encode()).hexdigest()[:14]
        for view in (initial, supplementary):
            for role, frames in (('reference', ref), ('candidate', rendered)):
                for state in STATES:
                    path = out / f'native/{cid}/{view}/{role}_{state}.png'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(frames['frames'][state][view])
        program_path = f'private/programs/{cid}.json'
        _json(out / program_path, to_dict(program.spec))
        rows['public'].append({'case_id': cid, 'initial_view': initial,
            'supplementary_view': supplementary, 'states': list(STATES),
            'native_root': f'native/{cid}', 'image_size': [256, 256],
            'status': 'pending_human_review'})
        rows['gold'].append({'case_id': cid, 'asset_id': reference.spec.asset_id,
            'config': item['config'], 'difficulty': 'hard_provisional',
            'screening_policy': POLICY['version'], 'kind': kind,
            'expected_status': 'pass' if label is None else 'fail',
            'defect_id': None if label is None else label.defect_id,
            'joint_id': joint_id, 'parts': [] if label is None else list(label.parts),
            'parameters': {} if label is None else label.params,
            'seed': seed, 'state_values': values, 'candidate_sha': program.sha,
            'reference_sha': reference.sha, 'program': program_path,
            'initial_view': initial, 'supplementary_view': supplementary,
            'route': chosen['route'], 'source': source,
            'training_eligibility': 'pending_human_review'})
        rows['checks'].append({'case_id': cid, **check})
        rows['reviews'].append(_review(cid))
    return rows


def page(out, public, name='review.html'):
    """Emit a standalone offline page containing only constrained public fields."""
    out = Path(out)
    if Path(name).name != name or not name.endswith('.html'):
        raise ValueError('Review page name must be a plain HTML filename')
    cases = []
    for row in public:
        cid, first, second = (row[k] for k in ('case_id', 'initial_view', 'supplementary_view'))
        if (not re.fullmatch(r'hp-[0-9a-f]{14}', cid) or first == second
                or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', v) for v in (first, second))
                or row['states'] != list(STATES) or row['native_root'] != f'native/{cid}'):
            raise ValueError('Invalid opaque case identity, fixed views or native paths')
        cases.append({'case_id': cid, 'initial_view': first, 'supplementary_view': second,
                      'native_root': f'native/{cid}'})
    if not cases or len({row['case_id'] for row in cases}) != len(cases):
        raise ValueError('Review requires a nonempty unique case collection')
    payload = json.dumps(cases, ensure_ascii=False, separators=(',', ':'))
    package = hashlib.sha256(payload.encode()).hexdigest()[:20]
    document = (ASSETS / 'review.template.html').read_text(encoding='utf-8')
    document = document.replace('{{CASES_JSON}}', payload).replace('{{PACKAGE}}', package)
    document = document.replace('{{STYLE}}', (ASSETS / 'review.css').read_text())
    document = document.replace('{{SCRIPT}}', (ASSETS / 'review.js').read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(document, encoding='utf-8')
