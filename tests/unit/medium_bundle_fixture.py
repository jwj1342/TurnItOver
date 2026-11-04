"""Deterministic, wholly synthetic 75-object fixture; never a benchmark result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import zlib

from scripts.easy_release_archive import sha256
from scripts.medium_bundle_history import RAW_KEYS, history_files
from scripts.medium_bundle_inventory import write_manifest
from scripts.prepare_medium_review import prepare_review
from scripts.validate_medium_bundle import EXPECTED_CONFIGS, KINDS, LEDGERS, STATES


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def png(color=32):
    chunk = lambda name, data: struct.pack('>I', len(data)) + name + data + struct.pack('>I', zlib.crc32(name + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 256, 256, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + bytes([color]) * 768) * 256)) + chunk(b'IEND', b''))


def build_bundle(root):
    gold, public, human, checks, programs = [], [], [], [], []
    data, index = png(), 0
    for config, count in EXPECTED_CONFIGS.items():
        for _ in range(count):
            source = {k: f'synthetic-{k}-{index}' for k in
                      ('source_identity', 'model_id', 'archive_sha256', 'original_spec_sha256', 'original_geometry_fingerprint')}
            source['split'] = 'train'
            for clean in (True, False):
                cid = f'ms-{index * 2 + (not clean):014x}'
                reference_sha = f'{index:016x}'
                candidate_sha = reference_sha if clean else f'{index + 1000:016x}'
                kind = 'clean' if clean else KINDS[config]
                row = dict(case_id=cid, asset_id=f'synthetic-object-{index}', config=config, kind=kind,
                    source=source.copy(), difficulty='medium_provisional', training_eligibility='pending_human_review',
                    screening_policy='medium-strength-v3', expected_status='pass' if clean else 'fail',
                    reference_sha=reference_sha, candidate_sha=candidate_sha, state_values={'rest': 0, 'mid': .1, 'limit': .2},
                    program=f'private/programs/{cid}.json', parts=[] if clean else ['moving'],
                    defect_id=None if clean else ('structure.part_offset' if kind == 'part_offset' else 'kinematics.' + kind))
                gold.append(row)
                programs.append(dict(case_id=cid, source_path=row['program'],
                    sha256=hashlib.sha256(('synthetic program ' + cid).encode()).hexdigest(), size=1,
                    candidate_sha=candidate_sha, reference_sha=reference_sha))
                p = dict(case_id=cid, board=f'boards/{cid}.png', input_preview=f'inputs/{cid}.png',
                         images=[f'frames/{cid}/{i}.png' for i in range(4)], states=list(STATES),
                         input_states=['rest', 'limit'], image_size=[256, 256], view='synthetic', status='pending_human_review')
                public.append(p)
                human.append(dict(case_id=cid, reviewer=None, reviewed_at=None, status='pending_human_review',
                                  human_judgment={'verdict': None, 'evidence_states': []}))
                invariants = [dict(parts=['moving'], state=s, passed=clean)
                              for s in ('rest', 'joint:target:mid', 'joint:target:limit')]
                checks.append(dict(case_id=cid, candidate_sha=candidate_sha, reference_sha=reference_sha,
                    code_correct=clean, privileged=True, used_by_student=False, checks=[
                    dict(checker_id='geometry.alignment', passed=clean, invariants=invariants),
                    dict(checker_id='runtime.budget', passed=True, invariants=[{'passed': True}])]))
                paths = [p['board'], p['input_preview'], *p['images']]
                paths += [f'native/{cid}/{role}_{state}.png' for role in ('reference', 'candidate') for state in STATES]
                for name in paths:
                    file = root / name
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_bytes(data)
            index += 1
    for name, rows in [('private/gold.jsonl', gold), ('review_manifest.jsonl', public),
                       ('human_review.jsonl', human), ('private/checks.jsonl', checks)]:
        write_rows(root / name, rows)
    (root / 'train_manifest.jsonl').write_text('')
    (root / 'private/train_targets.jsonl').write_text('')
    ledger, dispositions = [], []
    approved = [g for g in gold if g['kind'] != 'clean'][:11]
    for number, decisions in [(1, [*[(g, 'medium') for g in approved[:8]], (None, 'clean'), (None, 'clean')]),
                              (2, [*[(g, 'medium') for g in approved[8:]], *[(None, 'too_obvious')] * 3])]:
        folder = root / f'private/human_review_history/round-{number:04d}'
        raw = []
        for g, decision in decisions:
            case = g or dict(case_id=f'ms-{10000 + len(ledger):014x}', asset_id=f'synthetic-excluded-{len(ledger)}',
                             candidate_sha='a' * 16, reference_sha='b' * 16, source={'source_identity': 'excluded'})
            original = dict(case_id=case['case_id'], reviewer='human', decision=decision,
                            notes='synthetic fixture only', reviewed_at='2026-10-24T05:00:00Z')
            row = dict(original, asset_id=case['asset_id'], reviewer_type='human', scope='spot_visibility_only',
                       source_case={k: case[k] for k in ('asset_id', 'candidate_sha', 'reference_sha')})
            row['source_case']['source_identity'] = case['source']['source_identity']
            raw.append(original)
            ledger.append(row)
            dispositions.append(dict(case_id=row['case_id'], asset_id=row['asset_id'], reviewer_type='human',
                human_decision=decision, notes=row['notes'], disposition='spot_medium_observed_full_diagnostics_pending'
                if decision == 'medium' else 'excluded_by_human_spot_review'))
        write_json(folder / 'raw.json', raw)
        write_json(folder / 'source_manifest.json', {'synthetic_history_round': number})
        page = 'source_spot_check_10.json' if number == 1 else 'source_review_page.html'
        key = 'source_spot_sha256' if number == 1 else 'source_review_page_sha256'
        if number == 1:
            write_json(folder / page, {'case_ids': [r['case_id'] for r in raw]})
        else:
            (folder / page).write_text(''.join(f'<section data-id="{r["case_id"]}"></section>' for r in raw))
        for row in ledger[-len(raw):]:
            row['source_manifest_sha256'] = sha256(folder / 'source_manifest.json')
            row[key] = sha256(folder / page)
        write_rows(root / 'private/human_spot_review.jsonl', ledger)
        write_rows(root / 'private/human_spot_dispositions.jsonl', dispositions)
        write_json(folder / 'import.json', dict(source_manifest_sha256=sha256(folder / 'source_manifest.json'),
            raw_sha256=sha256(folder / 'raw.json'), **{key: sha256(folder / page)},
            ledger_sha256=sha256(root / 'private/human_spot_review.jsonl'),
            dispositions_sha256=sha256(root / 'private/human_spot_dispositions.jsonl')))
    write_json(root / 'source/summary.json', {'objects': 75, 'cases': 150, 'fixture': True})
    write_json(root / 'source/automatic-checks.json', dict(status='automatic_checks_passed_pending_human_review',
        objects=75, cases=150, selection_replayed_objects=75, generation_inputs_verified=True,
        selection_policy='medium-selection-tightening-v1', human_spot_medium_cases=11, human_excluded_objects=5))
    write_json(root / 'source/tightening_policy.json', {'version': 'medium-selection-tightening-v1'})
    originals = LEDGERS | history_files()
    originals |= {p.relative_to(root).as_posix() for p in root.rglob('*.png')}
    files = {name: sha256(root / name) for name in originals}
    files.update({'summary.json': sha256(root / 'source/summary.json'),
                  'private/verification/automatic-checks.json': sha256(root / 'source/automatic-checks.json'),
                  'private/tightening_policy.json': sha256(root / 'source/tightening_policy.json')})
    files.update({r['source_path']: r['sha256'] for r in programs})
    write_json(root / 'source/manifest.json', {'policy': {'version': 'medium-strength-v3'}, 'files': files})
    identity = sha256(root / 'source/manifest.json')
    write_json(root / 'source/metadata.json', dict(version=1, source_manifest_sha256=identity,
        source_code_commit='1' * 40, source_dirty=True, generation_policy='medium-strength-v3',
        selection_policy='medium-selection-tightening-v1', archive_availability='local_custody',
        benchmark_frozen=False, training_ready=False))
    write_json(root / 'construction_registry.json', dict(availability='local_custody',
        source_manifest_sha256=identity, programs=programs))
    mapping = prepare_review(root, root.parent / (root.name + '-review'), 'synthetic-medium-unit-only')
    shutil.move(str(root.parent / (root.name + '-review')), root / 'review_bundle')
    write_rows(root / 'private/review_mapping.jsonl', mapping)
    (root / 'README.md').write_text('Synthetic unit fixture; programs remain in local custody.\n')
    write_manifest(root)
    return root
