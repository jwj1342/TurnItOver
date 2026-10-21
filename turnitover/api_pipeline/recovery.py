"""Recover only a failed final API decision from unchanged, already-real evidence."""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

from turnitover.core.actions import Observation
from turnitover.core.jsonio import write_json_atomic
from turnitover.core.program import ObjectProgram
from turnitover.core.serde import from_dict
from turnitover.repair.iterative import IterativeConfig, run_iterative
from turnitover.repair.loop import usage_tokens
from turnitover.telemetry import now_iso
from turnitover.verifier.contracts import Context, Verdict, parse_decision
from turnitover.verifier.policies import VisionPolicy
from turnitover.verifier.report import write_report
from turnitover.verifier.runner import VerifyConfig

from .paired import file_sha


def recover_final(prepared, source, output, render, views, api):
    """No new Active evidence or source is generated. Old failures remain in a new copy.

    Limited to an exhausted-budget, post-repair Active final decision interrupted
    by transport failure, before Fixed started. This is not general run resumption.
    """
    prepared, source, output = (Path(p).resolve() for p in (prepared, source, output))
    report = json.loads((source / 'result.json').read_text())
    if report['status'] != 'incomplete' or len(report['initials']) != 1:
        raise ValueError('Recovery requires one incomplete pair')
    if report['model'] != api.public_config:
        raise ValueError('Recovery sampling/transport-timeout configuration changed')
    if file_sha(prepared / 'manifest.json') != report['prepared_manifest_sha']:
        raise ValueError('Prepared bundle changed')
    for name, digest in json.loads((prepared / 'manifest.json').read_text())['files'].items():
        if file_sha(prepared / name) != digest:
            raise ValueError('Prepared input changed')
    if not report.get('runtime_files') or any(file_sha(render.harness_dir / name) != digest
            for name, digest in report['runtime_files'].items()):
        raise ValueError('Recovery runtime changed')
    case_id = next(iter(report['initials']))
    active = source / case_id / 'active'
    arm = json.loads((active / 'result.json').read_text())
    if (source / case_id / 'fixed').exists() or arm['model_calls']['visual_revision'] != report['visual_revisions']:
        raise ValueError('Recovery requires exhausted repairs and no started Fixed arm')
    round_id = arm['rounds'][-1]['round']
    verifier_relative = Path(f'round-{round_id:03d}/verifier')
    old_verifier = active / verifier_relative
    verification = json.loads((old_verifier / 'result.json').read_text())
    if (verification.get('termination') != 'model_error' or verification['spent'] != report['observation_budget']
            or len(verification['trajectory']) != verification['spent']):
        raise ValueError('Only a transport-failed final decision with full real history can be recovered')
    old_ledger_path = source.with_name(source.name + '-api') / 'ledger.json'
    old_ledger = json.loads(old_ledger_path.read_text())
    if api.returned_model != old_ledger['returned_model']:
        raise ValueError('Recovery returned-model identity differs')
    last = old_ledger['calls'][-1]
    error = json.loads((old_ledger_path.parent / last['directory'] / 'error.json').read_text())
    if last['scope'] != f'{case_id}/active/judge' or not (
            error.get('http_code') == 524 or error['type'] == 'TimeoutError'):
        raise ValueError('Only a documented 524/timeout transport failure is recoverable here')
    failed_dir = old_verifier / verification['model_calls'][-1]['path']
    expected_prompt = (failed_dir / 'prompt.txt').read_text()
    expected_images = [file_sha(old_verifier / p) for p in json.loads((failed_dir / 'images.json').read_text())]
    wire_dir = old_ledger_path.parent / last['directory']
    wire = json.loads((wire_dir / 'request.json').read_text())
    content = wire['messages'][0]['content']
    wire_images = [hashlib.sha256(base64.b64decode(c['image_url']['url'].split(',', 1)[1])).hexdigest()
                   for c in content if c['type'] == 'image_url']
    wire_text = [c['text'] for c in content if c['type'] == 'text']
    if (wire_text != [expected_prompt] or wire_images != expected_images
            or (wire_dir / 'prompt.txt').read_text() != expected_prompt
            or wire['model'] != api.config.model or wire['max_tokens'] != api.config.max_tokens
            or wire.get('temperature') != api.temperature):
        raise ValueError('Recovery evidence/config differs from the original wire request')
    history = tuple(from_dict(Observation, item) for item in verification['trajectory'])
    context = from_dict(Context, verification['context'])
    if ObjectProgram((old_verifier / 'candidate.ts').read_text()).sha != verification['program_sha']:
        raise ValueError('Recovered candidate changed')
    if arm['final_program']['sha'] != verification['program_sha']:
        raise ValueError('Recovery final manifest differs')
    if ObjectProgram((active / 'initial.ts').read_text()).sha != report['initials'][case_id]:
        raise ValueError('Recovery Active initial differs')
    if ObjectProgram((active / 'final.ts').read_text()).sha != verification['program_sha']:
        raise ValueError('Recovery final program differs from recorded verifier candidate')
    if ObjectProgram((source / case_id / 'initial.ts').read_text()).sha != report['initials'][case_id]:
        raise ValueError('Recovery initial program changed')
    provenance = {str(p.relative_to(source)): file_sha(p) for p in source.rglob('*') if p.is_file()}
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    if any(file_sha(output / name) != digest for name, digest in provenance.items()):
        raise ValueError('Copied continuation history differs')
    report.update(status='recovering', continuation={
        'source_run': source.name, 'source_result_sha256': file_sha(source / 'result.json'),
        'created_at': now_iso(), 'kind': 'final_decision_transport_recovery',
        'new_active_observations': 0, 'prior_termination': verification['termination'],
        'copied_history_sha256': provenance})
    report.pop('error', None)
    report.pop('error_type', None)
    write_json_atomic(output / 'result.json', report)
    try:
        verifier_dir = output / case_id / 'active' / verifier_relative
        def guarded_call(prompt, images):
            if prompt != expected_prompt or [file_sha(p) for p in images] != expected_images:
                raise ValueError('Recovery final prompt or images differ from the failed request')
            return api.scoped(f'{case_id}/active/judge/recovery')(prompt, images)
        refs = [verifier_dir / r['path'] for r in verification['references']]
        policy = VisionPolicy(guarded_call, verifier_dir, refs, 'active', exact_budget=True)
        policy.calls = list(verification['model_calls'])
        raw = policy.decide(context, history, 0)
        verdict = parse_decision(raw, context, {o.step for o in history}, final_only=True)
        if not isinstance(verdict, Verdict):
            raise ValueError('Recovery did not produce a final verdict')
        verification.update(status='complete', termination='budget_exhausted',
            verdict=dataclasses.asdict(verdict), model_calls=policy.calls, finished_at=now_iso(),
            continuation=report['continuation'], error_type=None, error_message=None)
        write_json_atomic(verifier_dir / 'result.json', verification)
        write_json_atomic(verifier_dir / 'feedback.json', {'verdict': verification['verdict'],
            'termination': verification['termination'], 'program_sha': verification['program_sha'],
            'evidence_file': 'trajectory.jsonl'})
        write_report(verifier_dir, verification, history)
        arm['rounds'][-1]['verifier'].update(status='complete', termination='budget_exhausted',
                                            verdict=verification['verdict'])
        arm['model_calls']['judge'] += 1
        arm['total_tokens'] += usage_tokens(policy.calls[-1]['usage'])
        for key in ('error', 'error_type', 'phase'):
            arm.pop(key, None)
        arm.update(status={'pass': 'accepted', 'uncertain': 'verification_uncertain',
                           'fail': 'max_visual_revisions'}[verdict.status], accepted=verdict.status == 'pass',
                   finished_at=now_iso(), termination='budget_exhausted', continuation=report['continuation'])
        write_json_atomic(output / case_id / 'active/result.json', arm)
        program = ObjectProgram((output / case_id / 'initial.ts').read_text())
        cfg = IterativeConfig(max_runtime_repairs=1, max_visual_revisions=report['visual_revisions'],
            verifier=VerifyConfig(budget=report['observation_budget'], mode='fixed', exact_budget=True),
            task=(prepared / 'task.txt').read_text())
        fixed = run_iterative((prepared / 'reference.png',), output / case_id / 'fixed', render, views, cfg,
            api.scoped(f'{case_id}/fixed/generator'), initial_program=program,
            judge_call=api.scoped(f'{case_id}/fixed/judge'), judge_model=api.config)
        if fixed['status'] in {'generation_failed', 'verification_error', 'render_failed'}:
            raise RuntimeError('Fixed failed; preserve the partial continuation without retry')
        if any(file_sha(render.harness_dir / name) != digest for name, digest in report['runtime_files'].items()):
            raise ValueError('Runtime changed while continuing the pair')
        gates = [output / case_id / mode / 'round-000/render-gate/attempt-000/render.png'
                 for mode in ('active', 'fixed')]
        if file_sha(gates[0]) != file_sha(gates[1]):
            raise ValueError('Recovered pair initial render differs')
        write_json_atomic(output / case_id / 'pair.json', {'order': ['active', 'fixed'],
            'arms': {'active': arm, 'fixed': fixed}, 'continuation': report['continuation']})
        report['pairs'][case_id] = {'order': ['active', 'fixed'], 'initial_inputs_equal': True,
            'initial_render_sha': file_sha(gates[0]), 'arms': {mode: {'status': r['status'],
                'accepted': r['accepted'], 'model_calls': r['model_calls'], 'total_tokens': r['total_tokens'],
                'rounds': len(r['rounds'])} for mode, r in (('active', arm), ('fixed', fixed))}}
        # Both ledgers are required: the failed attempt is never erased or charged as zero.
        paths = [old_ledger_path, api.output / 'ledger.json']
        report['cost_ledgers'] = [{'path': '../' + p.parent.name + '/' + p.name, 'sha256': file_sha(p)} for p in paths]
        report.update(status='complete', finished_at=now_iso())
    except Exception as exc:
        report.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json_atomic(output / 'result.json', report)
    return report
