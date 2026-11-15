"""Run matched RP29 Table 1 conditions with OpenHands and independent audits.

Without an independent audit_factory this collects trajectories only. Pixel
judge pass is never interpreted as independent repair success.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import tempfile
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, distributions, version
import json
from pathlib import Path
import subprocess
import sys

from turnitover.models.client import complete
from turnitover.file_lock import file_lock
from turnitover.models.config import model_config, read_environment
from turnitover.repair.frozen_manifest import _file, load_frozen_manifest
from turnitover.repair.table1_judges import CONDITIONS, ObservationPair, make_public_record
from turnitover.repair.table1_scoring import summarize_condition
from turnitover.repair.table1_scoring import checker_coverage, classify_exposure
from turnitover.repair.table1_runner import run_table1_repair


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_cases(manifest):
    manifest = Path(manifest).resolve()
    metadata = json.loads(manifest.read_text())
    if (metadata.get('status') != 'frozen' or metadata.get('split') != 'validation'
            or metadata.get('benchmark') != 'RP29 v2.1-pre.1'
            or metadata.get('counts') != {'validation_total': 120, 'defective': 109, 'excluded_clean': 11}
            or len(metadata.get('cases', [])) != 109):
        raise ValueError('Expected frozen RP29 v2.1 Validation 109-defect batch')
    loaded, seen = [], set()
    for row in metadata['cases']:
        path = _file(manifest.parent, row['manifest'], row['manifest_sha256'], 'case manifest')
        item, cases = load_frozen_manifest(path)
        if (item.get('benchmark') != metadata['benchmark'] or len(cases) != 1
                or cases[0]['id'] != row['id'] or row['id'] in seen):
            raise ValueError('Batch/case identity mismatch')
        seen.add(row['id'])
        loaded.append((path, cases[0]))
    if metadata.get('private_labels') is not None:
        private_path = _file(manifest.parent, metadata['private_labels'],
                             metadata.get('private_labels_sha256'), 'private label roster')
        private_rows = [json.loads(line) for line in private_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(private_rows) != len(loaded) or {row['id'] for row in private_rows} != seen:
            raise ValueError('Private label/case roster mismatch')
        private = {row['id']: row for row in private_rows}
        for _, case in loaded:
            row = private[case['id']]
            if not row.get('labels'):
                raise ValueError('Defective case requires a nonempty private label roster')
            case['private_labels'] = row['labels']
            case['private_scoring_scopes'] = row.get('scoring_scopes')
            case['private_exposure'] = row.get('evidence', {})
            case['private_exposure_history'] = row.get('history', [])
    return metadata, loaded


def native_replay(manifest, native_root, output, *, python=None):
    """Cache only complete fresh native measurements bound to current source SHA."""
    cache = {}
    def replay(source):
        key = hashlib.sha256(source.encode()).hexdigest()
        if key not in cache:
            candidate = output / (key + '.ts')
            candidate.write_text(source, encoding='utf-8')
            folder = output / ('native-' + key)
            result = subprocess.run([str(python or sys.executable), str(Path(__file__).with_name('rp29_native_replay.py')),
                '--native-root', str(native_root), '--manifest', str(manifest),
                '--candidate', str(candidate), '--output', str(folder)], capture_output=True, text=True,
                timeout=300, encoding='utf-8')
            if result.returncode:
                raise RuntimeError('Native replay failed: ' + result.stderr[-2000:])
            evidence = json.loads((folder / 'replay.json').read_text())
            if (evidence.get('status') != 'complete'
                    or evidence.get('program_sha256', {}).get('candidate') != key):
                raise ValueError('Incomplete or stale native replay')
            cache[key] = evidence
        return cache[key]
    return replay


def run_case(manifest, case, native_root, output, generator, condition, rounds,
             calibration, audit_factory=None, replay_factory=native_replay, *,
             harness='openhands', max_agent_steps=None, judge_model_call=None,
             native_python=None, gold_factory=None):
    output.mkdir(parents=True, exist_ok=False)
    replay = (native_replay(manifest, native_root, output, python=native_python)
              if replay_factory is native_replay else replay_factory(manifest, native_root, output))
    gold_call = gold_factory(case, replay) if gold_factory and condition == 'gold' else None
    audit = (audit_factory(case, replay) if audit_factory else
             lambda source: {'passed': None, 'scope': 'unscored_trajectory',
                             'program_sha256': replay(source)['program_sha256'],
                             'reason': 'Independent full-defect checker unavailable'})
    def observe(source, folder):
        evidence = replay(source)
        pairs = [ObservationPair(p['state'], Path(p['reference']), Path(p['candidate']))
                 for p in evidence['pairs']]
        runtime = evidence.get('runtime', [])
        context = case.get('public_context')
        return {'pairs': pairs, 'runtime': runtime, 'gate': evidence.get('gate'),
                'public_record': make_public_record(pairs, runtime, context, case_id=case['id'],
                                                   view_id=case.get('view'))
                                 if context and len(pairs) == 3 else None}
    try:
        if harness == 'openhands':
            from turnitover.repair.openhands_runner import run_openhands_repair
            result = run_openhands_repair(case['initial_program'].read_text(encoding='utf-8'),
                generator, observe, audit, output / 'repair', condition=condition,
                max_rounds=rounds, max_agent_steps=max_agent_steps,
                reference_image=case['reference_image'], calibration=calibration,
                judge_model_call=judge_model_call, gold_call=gold_call,
                public_task={'description': case.get('public_task', ''),
                    'runtime_requirements': case.get('task_runtime_requirements', [])})
        elif harness == 'legacy':
            result = run_table1_repair(case['initial_program'].read_text(encoding='utf-8'), generator, observe, audit,
                output / 'repair', condition=condition, max_rounds=rounds,
                reference_image=case['reference_image'], calibration=calibration,
                judge_model_call=judge_model_call,
                public_task={'description': case.get('public_task', ''),
                         'runtime_requirements': case.get('task_runtime_requirements', [])},
                allow_unscored=True)
        else:
            raise ValueError('Unknown coding harness')
    except Exception as exc:
        result_path = output / 'repair/result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            'status': 'incomplete', 'total_tokens': 0, 'error_type': type(exc).__name__}
    write(output / 'summary.json', result)
    return result


def execution_hashes(native_root):
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), Path(__file__).with_name('rp29_native_replay.py').resolve(),
             Path(__file__).with_name('rp29_private_readback.js').resolve()]
    files += [root / name for name in (
        'turnitover/repair/table1_runner.py', 'turnitover/repair/table1_judges.py',
        'turnitover/repair/openhands_runner.py', 'turnitover/repair/table1_scoring.py',
        'turnitover/repair/native_evaluator.py', 'turnitover/verifier/gt_validator.py',
        'turnitover/repair/native_properties.py',
        'turnitover/repair/native_geometry.py', 'turnitover/repair/native_scopes.py',
        'docs/native-scoring-provenance.json',
        'requirements-openhands.txt',
        'turnitover/file_lock.py',
        'turnitover/repair/loop.py', 'turnitover/repair/patches.py',
        'turnitover/repair/frozen_manifest.py', 'turnitover/models/client.py',
        'turnitover/models/config.py', 'turnitover/models/request_control.py',
        'turnitover/models/streaming.py',
        'turnitover/verifier/contracts.py',
        'turnitover/oracle/observation_inference.py', 'turnitover/oracle/observation_inputs.py',
        'turnitover/oracle/supervision_validation.py')]
    for folder in ('turnitover/checkers', 'turnitover/taxonomy'):
        files += [p for p in (root/folder).rglob('*')
                  if p.is_file() and p.suffix in {'.py', '.yaml'}]
    for folder in ('turnitover/core', 'turnitover/render', 'web/dist', 'configs'):
        files += [p for p in (native_root/folder).rglob('*')
                  if p.is_file() and p.suffix in {'.py', '.js', '.html', '.yaml'}]
    return {str(p): digest(p) for p in sorted(set(files))}


def environment_snapshot(python=None):
    """Record the renderer venv too when SDK and native replay are separated."""
    if python is None or Path(python).resolve() == Path(sys.executable).resolve():
        return {'python': sys.version,
                'packages': {d.metadata['Name']: d.version for d in distributions()}}
    script = ('import json,sys; from importlib.metadata import distributions; '
              'print(json.dumps({"python":sys.version,"packages":'
              '{d.metadata["Name"]:d.version for d in distributions()}}))')
    result = subprocess.run([str(python), '-c', script], capture_output=True,
                            encoding='utf-8', timeout=30, check=True)
    return json.loads(result.stdout)


def _run(manifest, native_root, output, env, *, rounds=None, conditions=('none', 'pixel'),
        audit_factory=None, case_runner=run_case, collect_unscored=False, preflight_only=False,
        harness='openhands', max_agent_steps=None, native_python=None, judge_envs=None,
        gold_factory=None, repetition=None, evaluator_module=None):
    manifest, native_root, output = [Path(p).resolve() for p in (manifest, native_root, output)]
    metadata, cases = load_cases(manifest)
    evaluator = None
    contract = None
    if evaluator_module:
        evaluator = importlib.import_module(evaluator_module)
        audit_factory = evaluator.make_audit
        gold_factory = getattr(evaluator, 'make_gold', None)
        contract = getattr(evaluator, 'SCORING_CONTRACT', None)
        if not isinstance(contract, dict):
            raise ValueError('Evaluator module requires a SCORING_CONTRACT object')
    if (not conditions or len(set(conditions)) != len(conditions)
            or set(conditions) - set(CONDITIONS)
            or rounds is not None and (type(rounds) is not int or rounds < 1)):
        raise ValueError('Unique Table 1 conditions and positive rounds required')
    if harness not in {'openhands', 'legacy'}:
        raise ValueError('Unknown coding harness')
    if harness == 'legacy' and set(conditions) - {'none', 'pixel', 'zero-shot-single', 'zero-shot-sweep'}:
        raise ValueError('Full Table 1 matrix requires OpenHands')
    blockers = []
    scoring_environment = None
    if evaluator is not None and callable(getattr(evaluator, 'check_environment', None)):
        scoring_environment = evaluator.check_environment()
        if not scoring_environment.get('ready'):
            blockers.append('Independent scoring environment not ready: ' + json.dumps(scoring_environment))
    native_files = ('turnitover/render/session.py', 'configs/views.yaml',
                    'web/dist/index.html', 'web/node_modules/.bin/esbuild')
    missing_native = [name for name in native_files if not (native_root / name).is_file()]
    if missing_native:
        blockers.append('Native replay environment is missing: ' + ', '.join(missing_native))
    if rounds is None:
        blockers.append('Freeze --rounds explicitly; paper repair round budget is not specified')
    if harness == 'openhands':
        try:
            if version('openhands-sdk') != '1.62.0':
                blockers.append('Install pinned openhands-sdk==1.62.0 in the agent environment')
        except PackageNotFoundError:
            blockers.append('Install pinned openhands-sdk==1.62.0 in the agent environment')
    if native_python is not None and not Path(native_python).is_file():
        blockers.append('Native replay Python executable does not exist')
    if not env.get('TIO_GENERATOR_MODEL'):
        blockers.append('Set an explicit frozen generator model ID')
    if harness == 'openhands' and (type(max_agent_steps) is not int or max_agent_steps < 1):
        blockers.append('Freeze --max-agent-steps separately from repair rounds')
    if not repetition:
        blockers.append('Set an explicit --repetition identifier; paper repair seed count is not frozen')
    if any(c not in {'none', 'pixel'} for c in conditions) and any(not case.get('public_context') for _, case in cases):
        blockers.append('Refreeze cases with public_context for structured judges')
    model_conditions = set(conditions) - {'none', 'pixel', 'gold'}
    if model_conditions - set(judge_envs or {}):
        blockers.append('Missing judge endpoint/checkpoint configs: ' + ', '.join(sorted(model_conditions - set(judge_envs or {}))))
    if 'gold' in conditions and gold_factory is None:
        blockers.append('Gold needs current-program independent privileged findings, not static original labels')
    if audit_factory is None:
        blockers.append('Full independent scoring unavailable')
    elif any(not case.get('private_labels') for _, case in cases):
        blockers.append('Independent scoring requires a hash-bound private per-label roster')
    coverage = checker_coverage(contract.get('supported_defects', []) if contract else [])
    if contract is None or contract.get('status') != 'frozen' or not contract.get('required_checks') or not coverage['complete']:
        blockers.append('Freeze the full 29-class independent checker contract, including regression checks')
    if contract is None or contract.get('exposure_status') != 'frozen':
        blockers.append('Freeze reviewed per-label exposure annotations; automatic projections are insufficient')
    calibration = None
    if 'pixel' in conditions:
        calibration_path = _file(manifest.parent, metadata.get('pixel_calibration'),
                                 metadata.get('pixel_calibration_sha256'), 'pixel calibration')
        if not calibration_path.is_relative_to(manifest.parent):
            raise ValueError('Calibration must belong to frozen input bundle')
        calibration = json.loads(calibration_path.read_text())
        from turnitover.repair.table1_judges import _threshold
        _threshold(calibration)
    if preflight_only:
        return {'status': 'inputs_validated', 'cases': len(cases), 'api_calls': 0,
                'scoring': 'independent_callback' if audit_factory else 'unscored_trajectory', 'success_rate': None,
                'resolved_rate': None, 'cost_to_success': None, 'harness': harness,
                'experiment_ready': not blockers, 'blockers': blockers,
                'checker_coverage': coverage, 'scoring_contract': contract,
                'scoring_environment': scoring_environment}
    if audit_factory is None and not collect_unscored:
        raise ValueError('Full independent scoring unavailable; --collect-unscored is required '
                         'to explicitly collect trajectories without Success/Resolved scores.')
    execution_blockers = [b for b in blockers if b != 'Full independent scoring unavailable'
                          and not (collect_unscored and (b.startswith('Freeze the full 29-class')
                                   or b.startswith('Freeze reviewed per-label')))]
    if execution_blockers and case_runner is run_case:
        raise ValueError('; '.join(execution_blockers))
    config = model_config('generator', env)
    config.validate()
    program_access_policy = None
    if harness == 'openhands' and case_runner is run_case:
        from turnitover.repair.openhands_runner import generator_llm, PROGRAM_ACCESS_POLICY
        generator_llm(config)  # No API call; reject incompatible settings before the queue.
        program_access_policy = dict(PROGRAM_ACCESS_POLICY)
    judge_configs = {condition: replace(model_config('judge', values), max_tokens=512) for condition, values in (judge_envs or {}).items()
                     if condition in model_conditions}
    for judge_config in judge_configs.values():
        judge_config.validate()
    run_config = {'benchmark': metadata['benchmark'], 'manifest_sha256': digest(manifest),
                  'code_sha256': execution_hashes(native_root),
                  'endpoint_sha256': hashlib.sha256(config.base_url.encode()).hexdigest(),
                  'native_root': str(native_root), 'conditions': list(conditions), 'rounds': rounds,
                  'harness': harness, 'max_agent_steps': max_agent_steps,
                  'program_access_policy': program_access_policy,
                  'scoring_environment': scoring_environment,
                  'native_python': str(native_python) if native_python else sys.executable,
                  'repetition': repetition,
                  'evaluator': {'module': evaluator_module, 'contract': contract,
                       'module_sha256': digest(evaluator.__file__)} if evaluator else None,
                  'agent_environment': environment_snapshot(),
                  'native_environment': environment_snapshot(native_python),
                  'judges': {name: {'config': value.public(),
                      'decoding': {'temperature': 0, 'max_tokens': 512},
                      'endpoint_sha256': hashlib.sha256(value.base_url.encode()).hexdigest()}
                      for name, value in judge_configs.items()},
                  'generator': config.public(), 'seed': None, 'seed_support': 'not_requested_or_verified',
                  'repetitions': 1, 'calibration_sha256': digest(calibration_path) if calibration else None,
                  'scoring': 'unscored_trajectory' if audit_factory is None else 'independent_callback'}
    comparison_protocol = {name: run_config[name] for name in (
        'manifest_sha256', 'code_sha256', 'harness', 'max_agent_steps', 'rounds',
        'calibration_sha256', 'evaluator', 'agent_environment', 'native_environment', 'native_python',
        'program_access_policy', 'scoring_environment')}
    comparison_sha = hashlib.sha256(json.dumps(comparison_protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != run_config:
            raise ValueError('Cannot resume with changed run configuration')
    elif any(output.iterdir()):
        raise ValueError('Nonempty output without frozen configuration')
    else:
        write(config_path, run_config)
    rows = []
    for condition in conditions:
        for path, case in cases:
            folder = output / condition / case['id']
            existing = folder / 'summary.json'
            if existing.exists():
                result = json.loads(existing.read_text())
            elif folder.exists():
                # A crash may have spent API tokens: preserve its directory; never silently retry.
                result_path = folder / 'repair/result.json'
                result = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'incomplete'}
                write(existing, result)
            else:
                try:
                    if case_runner is run_case:
                        judge_config = judge_configs.get(condition)
                        result = run_case(path, case, native_root, folder,
                            config if harness == 'openhands' else lambda prompt, images: complete(config, prompt, images),
                            condition, rounds, calibration, audit_factory,
                            harness=harness, max_agent_steps=max_agent_steps, native_python=native_python,
                            judge_model_call=(lambda prompt, images: complete(judge_config, prompt, images, temperature=0)) if judge_config else None,
                            gold_factory=gold_factory)
                    else:
                        result = case_runner(path, case, native_root, folder,
                            lambda prompt, images: complete(config, prompt, images), condition, rounds,
                            calibration, audit_factory)
                except Exception as exc:
                    folder.mkdir(parents=True, exist_ok=True)
                    result_path = folder / 'repair/result.json'
                    result = json.loads(result_path.read_text()) if result_path.exists() else {
                        'status': 'incomplete', 'error_type': type(exc).__name__}
                    write(existing, result)
            # Even a failure before the first audit must retain injected label
            # instances in Resolved denominators. These fields never enter prompts.
            if not result.get('audit_roster') and case.get('private_labels'):
                result = {**result, 'audit_roster': [
                    {'label_index': index, 'defect_id': label['defect_id'],
                     'parts': label.get('parts', []), 'resolved': None,
                     'exposure': classify_exposure(index, case.get('private_exposure', {}),
                                                   case.get('private_exposure_history', []))}
                    for index, label in enumerate(case['private_labels'])]}
            rows.append({**result, 'id': case['id'], 'condition': condition,
                         'generator': config.model, 'repetition': repetition,
                         'generator_config': {'config': run_config['generator'],
                                              'endpoint_sha256': run_config['endpoint_sha256']},
                         'comparison_protocol_sha256': comparison_sha,
                         'judge_config': run_config['judges'].get(condition)})
            write(output / 'progress.json', {'target': 109 * len(conditions), 'processed': len(rows),
                                             'complete': sum(r['status'] == 'complete' for r in rows)})
    summary = {'status': 'complete' if all(r['status'] == 'complete' for r in rows) else 'incomplete',
               'target': 109 * len(conditions), 'processed': len(rows), 'scoring': run_config['scoring'],
               'total_tokens': sum(r.get('total_tokens', 0) for r in rows),
               'success_rate': None, 'resolved_rate': None, 'cost_to_success': None,
               'results': rows, 'limitations': ['Missing independent scores are null, never pass or failure.',
                   'No automatic retry of interrupted or incomplete cases; costs may be incomplete after transport failures.']}
    summary['conditions'] = {condition: summarize_condition([row for row in rows if row['condition'] == condition])
                             for condition in conditions}
    if len(conditions) == 1:
        metrics = summary['conditions'][conditions[0]]
        summary['success_rate'] = metrics['success_rate']
        summary['resolved_rate'] = metrics['resolved']
        summary['cost_to_success'] = metrics['cost_to_success']
    write(output / 'summary.json', summary)
    return summary


def run(manifest, native_root, output, env, **kwargs):
    # A host-local lock prevents two queues spending tokens on the same output.
    key = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / ('rp29-batch-' + key + '.lock')
    with _output_lock(lock_path):
        return _run(manifest, native_root, output, env, **kwargs)


@contextmanager
def _output_lock(path):
    # Translate only acquisition failure; an error in the run is not a lock conflict.
    manager = file_lock(path, blocking=False)
    try:
        manager.__enter__()
    except OSError:
        raise RuntimeError('Another process owns this batch output') from None
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'native-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--preflight', action='store_true', help='Validate all frozen inputs without API calls')
    parser.add_argument('--collect-unscored', action='store_true', help='Explicitly allow incomplete independent scores')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=['none', 'pixel'])
    parser.add_argument('--harness', choices=('openhands', 'legacy'), default='openhands')
    parser.add_argument('--max-agent-steps', type=int)
    parser.add_argument('--native-python', type=Path)
    parser.add_argument('--repetition')
    parser.add_argument('--judge-env', action='append', default=[], metavar='CONDITION=PATH')
    parser.add_argument('--evaluator-module',
                        help='Native evaluator module; built-in reuses existing checks and reports gaps')
    args = parser.parse_args()
    if not args.preflight and args.env_file is None:
        parser.error('--env-file is required for execution')
    if not args.preflight and args.rounds is None:
        parser.error('--rounds must explicitly freeze the repair budget')
    summary = run(args.manifest, args.native_root, args.output,
                  read_environment(args.env_file) if args.env_file else {},
                  rounds=args.rounds, conditions=args.conditions,
                  harness=args.harness, max_agent_steps=args.max_agent_steps,
                  native_python=args.native_python, repetition=args.repetition,
                  evaluator_module=(args.evaluator_module or
                      ('turnitover.repair.native_evaluator' if args.harness == 'openhands' else None)),
                  judge_envs={name: read_environment(Path(path)) for name, path in
                              (value.split('=', 1) for value in args.judge_env)},
                  collect_unscored=args.collect_unscored, preflight_only=args.preflight)
    print(json.dumps(summary if args.preflight else {key: summary[key] for key in
                     ('status', 'target', 'processed', 'scoring', 'total_tokens')}, indent=2))
