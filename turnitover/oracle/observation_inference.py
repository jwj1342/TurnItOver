"""Shared observation-wise inference and deterministic case aggregation."""
from __future__ import annotations

from dataclasses import asdict
import json
import math

from turnitover.oracle.observation_inputs import (
    STATES, ObservationInput, build_observation_inputs,
)
from turnitover.verifier.contracts import Context, parse_decision, parse_json


def _aggregate(observations):
    findings, metadata = {}, {}
    for observation in observations:
        if not observation['valid']:
            continue
        for finding in observation['verdict']['findings']:
            key = (finding['defect_id'], tuple(sorted(set(finding['parts']))), finding['joint_id'])
            if key not in findings:
                findings[key] = {**finding, 'parts': list(key[1]), 'evidence_steps': []}
                metadata[key] = {'defect_id': key[0], 'parts': list(key[1]), 'joint_id': key[2],
                                 'evidence_observations': [], 'suggested_fixes': [], 'conflicts': []}
            target, provenance = findings[key], metadata[key]
            target['evidence_steps'] = sorted(set(target['evidence_steps']) | set(finding['evidence_steps']))
            if observation['observation_id'] not in provenance['evidence_observations']:
                provenance['evidence_observations'].append(observation['observation_id'])
            suggestion = {'observation_id': observation['observation_id'], 'suggested_fix': finding['suggested_fix']}
            if suggestion not in provenance['suggested_fixes']:
                provenance['suggested_fixes'].append(suggestion)
            fixes = list(dict.fromkeys(x['suggested_fix'] for x in provenance['suggested_fixes']))
            if len(fixes) > 1:
                provenance['conflicts'] = fixes
                target['suggested_fix'] = 'Conflicting observation suggestions: ' + json.dumps(provenance['suggested_fixes'])
    complete = (len(observations) == 3 and {o['observation_id'] for o in observations} == set(STATES)
                and all(o['valid'] and o['status'] == 'pass' for o in observations))
    status = 'fail' if findings else 'pass' if complete else 'uncertain'
    verdict = {'status': status, 'confidence': 0.0, 'findings': list(findings.values()),
               'summary': 'Deterministic aggregation of independent observation verdicts.',
               'limitations': ['Case pass requires all three observations to explicitly pass.'] if status == 'uncertain' else []}
    return verdict, list(metadata.values())


def infer_observationwise(record, images, model_call, withheld=()):
    """model_call(single_pair_record, pair_images) returns raw text or a metadata dict.

    No private target or exposure record is used for construction or aggregation.
    Invalid outputs remain in observations and prevent an unearned case pass.
    """
    observations, usage = [], {}
    for item in build_observation_inputs(record, images, withheld):
        row = {'observation_id': item.observation_id, 'step': item.step, 'valid': False,
               'status': 'uncertain', 'raw': None, 'error': None, 'usage': {}}
        try:
            response = model_call(item.record, item.images)
        except Exception as exc:
            row.update(error=str(exc), error_type=type(exc).__name__)
            observations.append(row)
            continue
        response = {'raw': response} if isinstance(response, str) else response
        if not isinstance(response, dict):
            response = {'raw': None}
        raw = response.get('raw')
        row['raw'] = raw
        row['usage'] = dict(response.get('usage', {}))
        for name in ('input_tokens', 'output_tokens', 'elapsed_ms'):
            if name in response:
                row[name] = response[name]
                row['usage'].setdefault(name, response[name])
        for name, value in row['usage'].items():
            if type(value) in (int, float) and math.isfinite(value):
                usage[name] = usage.get(name, 0) + value
        try:
            if not isinstance(raw, str):
                raise ValueError('Missing textual model output')
            payload = json.loads(item.record['messages'][0]['content'].rsplit('\n', 1)[-1])
            verdict = parse_decision(parse_json(raw), Context(**payload['context']),
                                     set(item.step_map), final_only=True)
            value = asdict(verdict)
            for finding in value['findings']:
                finding['parts'] = list(finding['parts'])
                finding['evidence_steps'] = sorted({item.step_map[x] for x in finding['evidence_steps']})
            value['findings'] = list(value['findings'])
            value['limitations'] = list(value['limitations'])
            row.update(valid=True, status=verdict.status, verdict=value)
        except (ValueError, TypeError) as exc:
            row['error'] = str(exc)
        observations.append(row)
    if 'total_tokens' not in usage and ('input_tokens' in usage or 'output_tokens' in usage):
        usage['total_tokens'] = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
    verdict, metadata = _aggregate(observations)
    return {'verdict': verdict, 'observations': observations, 'findings_metadata': metadata, 'usage': usage}
