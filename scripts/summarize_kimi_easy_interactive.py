"""Summarize a completed active pilot and independently cross-check its SFT export."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.easy_teacher_data import read_rows
from scripts.prepare_obvious_defect_review import sha256_path
from scripts.run_kimi_easy_interactive import DEFAULT_OUT
from turnitover.core.jsonio import write_json_atomic
from turnitover.verifier.contracts import parse_json, DecisionError


def call_usage(call):
    """Recover provider usage retained by the existing no-text error, without retries."""
    if call.get('usage'):
        return call['usage'], 'response_metadata'
    message = call.get('error_message', '')
    if 'usage=' in message:
        try:
            usage = json.loads(message.split('usage=', 1)[1])
            if isinstance(usage, dict) and usage:
                return usage, 'original_provider_error_message'
        except ValueError:
            pass
    return {}, 'unavailable'


def summarize(out):
    summary = json.loads((out / 'summary.json').read_text())
    results = read_rows(out / 'case_results.jsonl')
    assert len(results) == 18
    sft = out / 'sft'
    actions = read_rows(sft / 'train.actions.jsonl')
    diagnosis = read_rows(sft / 'train.diagnosis.jsonl')
    sources = {r['case_id']: r for r in results}
    gold = {r['id']: r for r in read_rows(out / 'private/gold.jsonl')}
    assert all(sources[r['episode_id']]['strict_correct'] and
               json.loads(r['messages'][1]['content'])['verdict']['status'] in ('pass', 'fail')
               for r in diagnosis)
    identities = lambda rows: sorted({r['data_identity']['asset_id'] for r in rows})
    summary['sft_assets'] = {'actions': identities(actions), 'diagnosis': identities(diagnosis)}
    summary['diagnosis_by_kind'] = dict(Counter(sources[r['episode_id']]['kind'] for r in diagnosis))
    summary['action_types'] = dict(Counter(json.loads(r['messages'][1]['content'])['action']['type'] for r in actions))
    summary['failure_reasons'] = dict(Counter(reason for r in results for reason in r['failure_reasons']))
    summary['trustworthy_uncertain'] = sum(r['trustworthy_uncertain'] for r in results)
    summary['strict_assets'] = sorted({r['asset_id'] for r in results if r['strict_correct']})
    summary['field_correct_by_kind'] = {kind: sum(
        r['kind'] == kind and all(r['field_matches'].values()) for r in results)
        for kind in ('clean', 'joint_axis', 'part_offset')}
    primary = Counter()
    for r in results:
        category = ('correct' if r['strict_correct'] else
                    'missing_legal_terminal' if not r['complete_episode'] else
                    'uncertain' if r['verdict'] == 'uncertain' else
                    'unsupported_evidence_citation' if all(r['field_matches'].values()) else
                    'false_pass_on_defect' if r['verdict'] == 'pass' and r['kind'] != 'clean' else
                    'wrong_type_or_localization')
        primary[category] += 1
    summary['primary_outcomes'] = dict(primary)
    image_rows, call_rows, raw_diagnoses = [], [], []
    acquisition_patterns = Counter()
    accounted = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    recovered_calls, unavailable_calls, reasoning_tokens = 0, 0, 0
    for result in results:
        folder = out / 'cases' / result['case_id']
        episode = json.loads((folder / 'episode.json').read_text())
        raw_match, raw_verdict = None, None
        if episode['model_calls']:
            raw_path = folder / episode['model_calls'][-1]['path'] / 'response.txt'
            if raw_path.is_file():
                try:
                    raw_verdict = parse_json(raw_path.read_text()).get('verdict')
                    if isinstance(raw_verdict, dict):
                        expected = gold[result['case_id']]['target']
                        key = lambda f: (f['defect_id'], f.get('joint_id'), tuple(sorted(f['parts'])))
                        raw_match = (raw_verdict.get('status') == expected['status'] and
                                     sorted(map(key, raw_verdict.get('findings', []))) ==
                                     sorted(map(key, expected['findings'])))
                except (DecisionError, KeyError, TypeError, AttributeError):
                    raw_match = False
        raw_diagnoses.append({'case_id': result['case_id'], 'kind': result['kind'],
                              'complete_episode': result['complete_episode'],
                              'raw_fields_correct': raw_match,
                              'raw_verdict': raw_verdict,
                              'not_an_export_approval': True})
        acquisition_patterns[' -> '.join(
            r['decision']['type'] + (':' + r['decision']['detent'] if 'detent' in r['decision'] else '')
            for r in episode['trajectory'] if r.get('decision_source') == 'teacher')] += 1
        before = dict(accounted)
        unknown_before = unavailable_calls
        for call in episode['model_calls']:
            call_dir = folder / call['path']
            images = json.loads((call_dir / 'images.json').read_text())
            usage, usage_source = call_usage(call)
            recovered_calls += usage_source == 'original_provider_error_message'
            unavailable_calls += usage_source == 'unavailable'
            inp = usage.get('input_tokens', usage.get('prompt_tokens', 0))
            output = usage.get('output_tokens', usage.get('completion_tokens', 0))
            accounted['input_tokens'] += inp
            accounted['output_tokens'] += output
            accounted['total_tokens'] += usage.get('total_tokens', inp + output)
            reasoning_tokens += (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
            call_rows.append({'case_id': result['case_id'], **call,
                              'accounted_usage': usage, 'usage_source': usage_source,
                              'prompt_sha256': sha256_path(call_dir / 'prompt.txt'),
                              'visible_request_sha256': sha256_path(call_dir / 'request.json'),
                              'response_sha256': sha256_path(call_dir / 'response.txt')
                              if (call_dir / 'response.txt').exists() else None})
            for index, image in enumerate(images):
                image_rows.append({'case_id': result['case_id'], 'call': call['path'], 'index': index,
                                   'image': image, 'sha256': sha256_path(folder / image)})
        write_json_atomic(folder / 'token_accounting.json', {
            'original_episode_usage': episode['token_usage'],
            'accounted_usage': {k: accounted[k] - before[k] for k in accounted},
            'unavailable_calls': unavailable_calls - unknown_before,
            'actual_total_tokens': (None if unavailable_calls > unknown_before else
                                    accounted['total_tokens'] - before['total_tokens']),
            'accounted_usage_is_known_subtotal': True,
            'source': 'saved response metadata and saved provider-error usage only; no new API calls'})
    assert len(call_rows) == summary['api_calls'] <= 144
    summary.setdefault('token_usage_original_episode_sum', summary['token_usage'])
    summary['token_usage'] = accounted
    summary['reasoning_tokens_included_in_output_tokens'] = reasoning_tokens
    summary['usage_recovered_from_original_error'] = recovered_calls
    summary['usage_unavailable_calls'] = unavailable_calls
    summary['token_usage_complete'] = unavailable_calls == 0
    summary['actual_total_tokens'] = None if unavailable_calls else accounted['total_tokens']
    summary['token_usage_semantics'] = 'Known subtotal only; unavailable call usage is not zero.'
    summary['teacher_selected_acquisition_patterns'] = dict(acquisition_patterns)
    summary['raw_field_correct_by_kind'] = {kind: sum(
        r['kind'] == kind and r['raw_fields_correct'] is True for r in raw_diagnoses)
        for kind in ('clean', 'joint_axis', 'part_offset')}
    for name, rows in [('api_calls.jsonl', call_rows), ('teacher_image_manifest.jsonl', image_rows),
                       ('raw_terminal_analysis.jsonl', raw_diagnoses)]:
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_json_atomic(out / 'summary.json', summary)
    table = ['# Per-case active-teacher results', '',
             '| case_id | asset | kind | complete | strict | verdict | calls | scope rest/mid/limit | reasons |',
             '|---|---|---|---|---|---|---:|---|---|']
    for r in results:
        coverage = '/'.join('yes' if r['scope_coverage'][s] else 'no' for s in ('rest', 'mid', 'limit'))
        table.append('| ' + ' | '.join(map(str, [r['case_id'], r['asset_id'], r['kind'],
            r['complete_episode'], r['strict_correct'], r['verdict'], r['api_calls'], coverage,
            '; '.join(r['failure_reasons']) or 'none'])) + ' |')
    (out / 'CASE_RESULTS.md').write_text('\n'.join(table) + '\n')
    write_json_atomic(out / 'artifact_manifest.json', {
        'files': {str(p.relative_to(out)): sha256_path(p) for p in sorted(out.rglob('*'))
                  if p.is_file() and p.name != 'artifact_manifest.json'}})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    summarize(parser.parse_args().out)
