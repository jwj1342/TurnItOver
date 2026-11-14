"""Matched-budget repair orchestration with private, non-controlling audits."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from turnitover.repair.loop import Feedback, propose_repair, usage_tokens
from turnitover.repair.table1_judges import evaluate_judge


def _write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def run_table1_repair(initial, generator, observe, audit, output, *, condition='none',
                      max_rounds=3, reference_image=None, judge_model_call=None,
                      calibration=None, public_task=None, allow_unscored=False):
    """Run one frozen case/condition; callbacks never share private audit data.

    observe(source, directory) returns matched ObservationPair objects. It is
    skipped for none. audit(source) returns at least a boolean passed field,
    and may include usable, score and private details. Audits never control
    acceptance, stopping or model-visible history. A valid textual patch is
    accepted even when its program is unusable; a subsequent render failure is
    an incomplete run rather than a hidden rollback. none always spends the
    entire call budget, including calls returning stop or invalid patches.
    allow_unscored accepts explicit passed=None for trajectory collection;
    unknown audits never set first_success_round or tokens_to_first_success.
    """
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError('max_rounds must be a positive integer')
    if condition not in {'none', 'pixel', 'zero-shot-single', 'zero-shot-sweep'}:
        raise ValueError('Unknown Table 1 condition')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source, memory, ledger, audits = initial, [], [], []
    report = {'version': 1, 'status': 'incomplete', 'condition': condition,
              'max_rounds': max_rounds, 'rounds': ledger, 'generator_calls': 0,
              'generator_tokens': 0, 'judge_calls': 0, 'judge_tokens': 0,
              'first_success_round': None, 'tokens_to_first_success': None,
              'initial_sha256': hashlib.sha256(initial.encode()).hexdigest(),
              'limitations': ['Private checker pass is not per-defect resolution.',
                             'Missing provider usage cannot establish exact token cost.']}
    (output / 'initial.ts').write_text(initial, encoding='utf-8')

    def total_tokens():
        return report['generator_tokens'] + report['judge_tokens']

    def check(folder, round_number):
        result = audit(source)
        if (not isinstance(result, dict) or 'passed' not in result
                or not (type(result['passed']) is bool
                        or allow_unscored and result['passed'] is None)):
            raise ValueError('Private audit must return a boolean passed field')
        _write(folder / 'audit.private.json', result)
        audits.append(result)
        if result['passed'] and report['first_success_round'] is None:
            report['first_success_round'] = round_number
            report['tokens_to_first_success'] = total_tokens()
        return result

    def judge(folder):
        pairs = [] if condition == 'none' else observe(source, folder)
        runtime = []
        if isinstance(pairs, dict):
            runtime, pairs = pairs.get('runtime', []), pairs['pairs']
        def call(prompt, images):
            (folder / 'judge.prompt.txt').write_text(prompt, encoding='utf-8')
            response = judge_model_call(prompt, images)
            report['judge_calls'] += 1
            report['judge_tokens'] += usage_tokens(response.usage)
            _write(folder / 'judge.model.json', asdict(response))
            return response
        result = evaluate_judge(condition, pairs, model_call=call if judge_model_call else None,
                                calibration=calibration, runtime=runtime)
        _write(folder / 'judge.json', result)
        return result

    def generate(prompt, images):
        response = generator(prompt, images)
        report['generator_calls'] += 1
        report['generator_tokens'] += usage_tokens(response.usage)
        return response

    try:
        check(output, 0)
        for index in range(max_rounds):
            folder = output / f'round-{index:03d}'
            folder.mkdir()
            verdict = judge(folder)
            if condition != 'none' and verdict['status'] == 'pass':
                report['termination'] = 'judge_pass'
                break
            public = {} if condition == 'none' else {'status': verdict['status'], **verdict['feedback']}
            if public_task is not None:
                public['task_requirements'] = public_task
            packet = Feedback(public, (Path(reference_image),) if reference_image else (), 0)
            proposal = propose_repair(source, generate, packet, folder / 'generator',
                                      previous_attempts=memory, remaining_calls=max_rounds-index)
            if proposal.outcome == 'applied':
                source = proposal.source
            record = {'round': index + 1, 'outcome': proposal.outcome, 'error': proposal.error,
                      'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                      'generator_usage': proposal.usage, 'judge_usage': verdict['usage'],
                      'cumulative_tokens': total_tokens()}
            ledger.append(record)
            memory.append({'round': index + 1, 'outcome': proposal.outcome,
                           'error': proposal.error, 'response': proposal.response})
            (folder / 'accepted.ts').write_text(source, encoding='utf-8')
            check(folder, index + 1)
            if condition != 'none' and proposal.outcome == 'stop':
                report['termination'] = 'generator_stop'
                break
        else:
            report['termination'] = 'round_limit'
        report.update(status='complete', final_passed=audits[-1]['passed'],
                      total_tokens=total_tokens(), completed_rounds=len(ledger))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), total_tokens=total_tokens(),
                      completed_rounds=len(ledger))
        raise
    finally:
        (output / 'final.ts').write_text(source, encoding='utf-8')
        _write(output / 'result.json', report)
    return report
