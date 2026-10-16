"""Scoped motion-pass audit and causal SFT protocol checks for the Easy pilot."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.easy_teacher_data import PROTOCOL, read_rows
from scripts.prepare_obvious_defect_review import sha256_path
from turnitover.core.jsonio import write_json_atomic
from turnitover.oracle.supervision import load_examples
from turnitover.teacher.audit import audit_episode
from turnitover.teacher.witnesses import same_state


def scope_coverage(episode, manifest, action_audit):
    """Evaluate an explicit public scope, not a hidden clean label."""
    rows = manifest['reference_protocol']['states']
    return {row['label']['detent']: any(
        a.get('bound') and a.get('reference_comparable')
        and a['state']['view'] == row['label']['view']
        and same_state(a['state']['joints'], row['state']) for a in action_audit)
        for row in rows}


def audit_case(folder: Path, prepared, gold, annotation, manifest):
    audit = audit_episode(folder, gold, annotation, manifest=manifest)
    episode = json.loads((folder / 'episode.json').read_text())
    independent = audit['independent_evidence_audit']
    coverage = scope_coverage(episode, manifest, independent['action_audit'])
    verdict = episode.get('verdict') or {}
    status = verdict.get('status')
    # The existing auditor recognizes only its static/runtime pass scopes. This
    # adapter adds exactly the predeclared selected-joint scope, with all visible
    # reference states covered by frozen-bound live observations.
    scoped_pass = (episode['context'] == prepared['context']
                   and episode.get('reference_protocol') == PROTOCOL
                   and all(coverage.values())
                   and manifest['checker']['code_correct']
                   and not verdict.get('findings'))
    if status == 'pass' and scoped_pass:
        independent['rejection_reasons'] = [r for r in independent['rejection_reasons']
            if r != 'pass_scope_not_supported:task_scope_not_limited_to_an_auditable_single_reference_rule']
        independent['accepted'] = not independent['rejection_reasons']
        independent['objective_evidence_complete'] = True
        independent['evidence_sufficient'] = True
    independent['scope_coverage'] = coverage
    independent['scope_pass_rule'] = PROTOCOL
    if not audit['teacher_outputs_legal']:
        independent['accepted'] = False
        independent['rejection_reasons'].append('teacher_output_not_legal')
    write_json_atomic(folder / 'audit.json', audit)
    expected = gold['target']
    expected_findings = {(f['defect_id'], f['joint_id'], tuple(sorted(f['parts']))) for f in expected['findings']}
    findings = verdict.get('findings', [])
    field_matches = {
        'verdict': status == expected['status'],
        'defect_id': sorted(f['defect_id'] for f in findings) == sorted(f['defect_id'] for f in expected['findings']),
        'joint_id': sorted(str(f['joint_id']) for f in findings) == sorted(str(f['joint_id']) for f in expected['findings']),
        'parts': sorted(sorted(f['parts']) for f in findings) == sorted(sorted(f['parts']) for f in expected['findings']),
    }
    actual_findings = {(f['defect_id'], f['joint_id'], tuple(sorted(f['parts']))) for f in findings}
    strict = (audit['pipeline_ok'] and independent['accepted'] and all(field_matches.values())
              and actual_findings == expected_findings and len(findings) == len(expected['findings']))
    # Uncertain is retained, never relabeled or silently exported as a diagnosis.
    export_audit = json.loads(json.dumps(audit))
    if status == 'uncertain':
        export_audit['independent_evidence_audit']['accepted'] = False
        export_audit['independent_evidence_audit']['rejection_reasons'].append('uncertain_retained_separately')
    write_json_atomic(folder / 'export_audit.json', export_audit)
    kind = 'clean' if not expected['findings'] else expected['findings'][0]['defect_id'].split('.')[-1]
    result = {'case_id': episode['episode_id'], 'asset_id': episode['data_identity']['asset_id'],
              'kind': kind, 'run_record_complete': episode.get('status') in ('complete', 'error'),
              'complete_episode': bool(episode.get('verdict')) and audit['teacher_outputs_legal'] and audit['pipeline_ok'],
              'termination': episode.get('termination'), 'verdict': status,
              'field_matches': field_matches, 'strict_correct': bool(strict),
              'evidence_sufficient': independent['evidence_sufficient'],
              'scope_coverage': coverage, 'localization': independent['localization'],
              'evidence_outcome': independent['diagnosis_outcome'],
              'trustworthy_uncertain': status == 'uncertain' and independent['accepted'],
              'failure_reasons': independent['rejection_reasons'],
              'observations': len(episode['trajectory']),
              'teacher_observations': sum(r.get('decision_source') == 'teacher' for r in episode['trajectory']),
              'api_calls': episode.get('real_api_requests', len(episode['model_calls'])),
              'token_usage': episode['token_usage'], 'billing_cost_rmb': None,
              'usage_incomplete_calls': sum(not r.get('usage') for r in episode['model_calls']),
              'episode_sha256': sha256_path(folder / 'episode.json')}
    write_json_atomic(folder / 'result.json', result)
    return result


def validate_export(root: Path):
    """Run the real loader and verify prompts, targets and image order byte-for-byte."""
    counts, assets, errors = {}, set(), []
    for split in ('train', 'validation'):
        for kind in ('actions', 'diagnosis', 'episodes'):
            loaded = list(load_examples(root, split, kind, require_formal_training=True))
            counts[f'{split}.{kind}'] = len(loaded)
            for row, images in loaded:
                folder = Path(row['source_episode'])
                episode = json.loads((folder / 'episode.json').read_text())
                decisions = [r for r in episode['decisions'] if r['valid']]
                decision = decisions[row['decision_index']]
                call = folder / episode['model_calls'][decision['call']]['path']
                expected_prompt = (call / 'prompt.txt').read_text()
                if row['messages'][0]['content'] != expected_prompt:
                    errors.append(f'prompt_changed:{row["id"]}')
                if json.loads(row['messages'][1]['content']) != decision['decision']:
                    errors.append(f'teacher_target_changed:{row["id"]}')
                source_images = json.loads((call / 'images.json').read_text())
                hashes = [sha256_path(folder / p) for p in source_images]
                if hashes != row['image_sha256'] or len(images) != len(source_images):
                    errors.append(f'image_protocol_changed:{row["id"]}')
                assets.add(row['data_identity']['asset_id'])
    report = {'status': 'passed' if not errors else 'failed', 'loader': counts,
              'independent_assets': len(assets), 'asset_ids': sorted(assets),
              'exact_teacher_student_prompt_target_image_parity': not errors, 'errors': errors,
              'old_117_rows_merged': False, 'training_started': False}
    write_json_atomic(root / 'loader_validation.json', report)
    if errors:
        raise ValueError(errors)
    return report
