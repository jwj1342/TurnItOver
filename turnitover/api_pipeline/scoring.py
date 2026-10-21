"""Independent post-hoc scoring. No scoring object is passed to a model call."""
from __future__ import annotations

import json
from pathlib import Path

from turnitover.checkers.base import Evidence
from turnitover.checkers.geometry import GeometryAlignmentChecker
from turnitover.core.actions import Action
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import from_dict, to_dict
from turnitover.policy.loop import execute_observation
from turnitover.render.session import ObservationSession
from turnitover.repair.loop import usage_tokens
from turnitover.verifier.gt_validator import validate_code

# Private diagnostic labels; no generation/verification entry imports this module.
TARGETS = {
    "c00": [],
    "c01": [{"defect_id": "structure.part_offset", "parts": ["drawer"], "joint_id": None}],
    "c02": [{"defect_id": "kinematics.joint_axis", "parts": ["door"], "joint_id": "door_hinge"}],
}


def quality(candidate, reference, render, views):
    result = validate_code(candidate, reference, render, views, 5000, 32)
    checks = result.get('checks', [])
    invariants = [item for check in checks for item in check.get('invariants', [])]
    result['invariant_pass_count'] = sum(item['passed'] for item in invariants)
    result['invariant_count'] = len(invariants)
    result['invariant_pass_fraction'] = (sum(item['passed'] for item in invariants) / len(invariants)
                                         if invariants else None)
    return result


def replay_witnesses(program, reference, trajectory, render, views):
    """Replay recorded actions in separate real browser loads, compare per-step geometry.

    This is geometric witness consistency, not a judge of language or image visibility.
    Only observations containing images may serve as visual defect witnesses.
    """
    states = []
    with ObservationSession(render, views) as session:
        session.load(reference, framing=None)
        for obs in trajectory:
            execute_observation(session, from_dict(Action, obs['action']), obs['step'], lambda *_: 'private')
            states.append(session.export_geometry())
        session.load(program, framing=None)
        rows = []
        for obs, gold in zip(trajectory, states, strict=True):
            execute_observation(session, from_dict(Action, obs['action']), obs['step'], lambda *_: 'private')
            candidate = session.export_geometry()
            actual_evidence = Evidence(program.sha, True, None, {'observed': candidate}, {}, {})
            gold_evidence = Evidence(reference.sha, True, None, {'observed': gold}, {}, {})
            check = to_dict(GeometryAlignmentChecker().check(actual_evidence, gold_evidence))
            failed = sorted({part for item in check['invariants'] if not item['passed']
                             for part in item.get('parts', [])})
            rows.append({'step': obs['step'], 'image_acquired': bool(obs.get('image_ref')),
                         'mismatched_parts': failed, 'geometry_check': check})
    return rows


def diagnosis_metrics(result, target, witnesses):
    all_findings = result['verdict'].get('findings', [])
    out_of_scope = [f for f in all_findings if f['defect_id'].startswith('material.')]
    findings = [f for f in all_findings if not f['defect_id'].startswith('material.')]
    predicted_types = {f['defect_id'] for f in findings}
    expected_types = {f['defect_id'] for f in target}
    expected_parts = {(f['defect_id'], p) for f in target for p in f['parts']}
    predicted_parts = {(f['defect_id'], p) for f in findings for p in f['parts']}
    witness_map = {w['step']: w for w in witnesses}
    citations = [step for f in findings for step in f['evidence_steps']]
    supported = []
    for finding in findings:
        match = next((t for t in target if t['defect_id'] == finding['defect_id'] and
                      set(t['parts']) == set(finding['parts']) and t['joint_id'] == finding['joint_id']), None)
        has_witness = bool(match) and any(
            witness_map.get(step, {}).get('image_acquired') and
            set(match['parts']) <= set(witness_map[step]['mismatched_parts'])
            for step in finding['evidence_steps'])
        supported.append(bool(has_witness))
    verdict = result['verdict']['status']
    valid = result.get('status') == 'complete'
    return {'verdict': verdict, 'detection_correct': (None if not target and out_of_scope and not findings else
                valid and verdict == ('fail' if target else 'pass')),
            'type_exact': valid and predicted_types == expected_types,
            'out_of_scope_findings': out_of_scope,
            'part_localization': {'tp': len(expected_parts & predicted_parts),
                                  'fp': len(predicted_parts - expected_parts),
                                  'fn': len(expected_parts - predicted_parts)},
            'joint_localization_exact': {f['joint_id'] for f in findings if f['joint_id']} ==
                                        {f['joint_id'] for f in target if f['joint_id']},
            'citation_count': len(citations),
            'citations_exist': all(s in witness_map for s in citations),
            'geometric_witness_consistent_findings': sum(supported), 'finding_count': len(findings),
            'observations': result['spent'], 'observation_time_ms': sum(
                obs.get('elapsed_ms', 0) for obs in result.get('trajectory', [])),
            'model_calls': len(result.get('model_calls', []))}


def controlled_label_valid(case_id, validation):
    """Reject injection labels if the generated ABI implementation changes their meaning."""
    if validation.get('status') != 'complete':
        return False
    checks = validation.get('checks', [])
    if any(not c['passed'] for c in checks if c['checker_id'] != 'geometry.alignment'):
        return False
    failed = [i for c in checks if c['checker_id'] == 'geometry.alignment'
              for i in c.get('invariants', []) if not i['passed']]
    if case_id == 'c00':
        return not failed
    parts = {p for i in failed for p in i.get('parts', [])}
    if case_id == 'c01':
        return parts == {'drawer'} and any(i['state'] == 'rest' for i in failed)
    return parts == {'door'} and bool(failed) and all(i['state'] != 'rest' for i in failed)


def score_pairs(prepared, run, render, views):
    """All API work finishes before private references and injection labels are opened."""
    prepared, run = Path(prepared), Path(run)
    reference = ObjectProgram((prepared / 'private/reference.ts').read_text())
    base = ObjectProgram((prepared / 'shared/final.ts').read_text())
    base_quality = quality(base, reference, render, views)
    summary = {'kind': 'paired_api_posthoc_score', 'base_quality': base_quality,
               'labels_valid': base_quality.get('code_correct') is True, 'pairs': {},
               'scope': 'One authored procedural object, controlled faults, single sample per arm.',
               'diagnostic_scope': 'Geometry, joint motion and runtime; material findings excluded and listed separately.',
               'evidence_metric': 'Geometric witness consistency is a necessary proxy; it does not establish natural-language correctness or image visibility.'}
    manifest = json.loads((run / 'result.json').read_text())
    from .paired import file_sha
    if file_sha(prepared / 'manifest.json') != manifest['prepared_manifest_sha']:
        raise ValueError('Scoring prepared bundle differs from the paired run')
    prepared_manifest = json.loads((prepared / 'manifest.json').read_text())
    for name, expected in prepared_manifest['files'].items():
        if file_sha(prepared / name) != expected:
            raise ValueError(f'Prepared input changed before scoring: {name}')
    for name, expected in manifest.get('runtime_files', {}).items():
        if file_sha(render.harness_dir / name) != expected:
            raise ValueError('Scoring harness differs from the recorded paired runtime')
    if manifest.get('status') != 'complete':
        raise ValueError('Only completed paired runs can be scored; preserve incomplete artifacts')
    records = manifest.get('cost_ledgers', [{'path': '../' + run.name + '-api/ledger.json'}])
    ledger_paths = [(run / r['path']).resolve() for r in records]
    if len(set(ledger_paths)) != len(ledger_paths):
        raise ValueError('Duplicate cost ledger would double count requests')
    ledger = {'calls': []}
    for record, path in zip(records, ledger_paths, strict=True):
        if not path.exists():
            raise ValueError('API ledger is required for cost scoring; missing cost is not zero')
        if record.get('sha256') and file_sha(path) != record['sha256']:
            raise ValueError('Continuation cost ledger changed')
        entries = json.loads(path.read_text())['calls']
        ledger['calls'].extend(entries)
    for case_id in manifest['pairs']:
        arms = {}
        for mode in ('active', 'fixed'):
            root = run / case_id / mode
            result = json.loads((root / 'result.json').read_text())
            rounds = []
            for round_record in result['rounds']:
                index = round_record['round']
                folder = root / f'round-{index:03d}'
                verifier = json.loads((folder / 'verifier/result.json').read_text())
                program = ObjectProgram((folder / 'verifier/candidate.ts').read_text())
                q = quality(program, reference, render, views)
                record = {'round': index, 'quality': q, 'verdict': verifier['verdict'],
                          'observations': verifier['spent'], 'actions': [o['action'] for o in verifier['trajectory']]}
                record['labels_valid'] = bool(summary['labels_valid'] and index == 0 and
                    controlled_label_valid(case_id, q))
                if record['labels_valid']:
                    witnesses = replay_witnesses(program, reference, verifier['trajectory'], render, views)
                    record['diagnosis'] = diagnosis_metrics(verifier, TARGETS[case_id], witnesses)
                    record['witnesses'] = witnesses
                rounds.append(record)
            final = ObjectProgram((root / 'final.ts').read_text())
            final_quality = quality(final, reference, render, views)
            passed = [r['round'] for r in rounds if r['quality'].get('code_correct')]
            calls = [c for c in ledger['calls'] if c['scope'].startswith(f'{case_id}/{mode}/')]
            costs = [c.get('provider_cost') for c in calls]
            api_cost = {'requests': len(calls), 'tokens': sum(usage_tokens(c['usage']) for c in calls),
                        'usage_reported_requests': sum(bool(c.get('usage')) for c in calls),
                        'requests_without_usage': sum(not bool(c.get('usage')) for c in calls),
                        'elapsed_ms': sum(c.get('elapsed_ms', 0) for c in calls),
                        'reported_cost_count': sum(v is not None for v in costs),
                        'provider_cost_total': sum(costs) if calls and all(type(v) in (int, float) for v in costs) else None}
            arms[mode] = {'api_cost': api_cost, 'status': result['status'], 'verifier_accepted': result['accepted'],
                          'rounds': rounds, 'final_quality': final_quality,
                          'first_correct_round': min(passed) if passed else None,
                          'visual_repair_calls': result['model_calls']['visual_revision'],
                          'runtime_repair_calls': result['model_calls']['runtime_repair'],
                          'tokens': result['total_tokens'], 'model_calls': result['model_calls'],
                          'total_observations': sum(r['observations'] for r in rounds)}
        summary['pairs'][case_id] = arms
    write_json_atomic(run / 'scores.private.json', summary)
    return summary
