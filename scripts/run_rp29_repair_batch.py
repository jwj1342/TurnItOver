"""Run matched RP29 Validation none/pixel trajectories, preserving unknown scores.

Without an independent audit_factory this collects trajectories only. Pixel
judge pass is never interpreted as independent repair success.
"""
from __future__ import annotations

import argparse
import fcntl
import tempfile
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from turnitover.models.client import complete
from turnitover.models.config import model_config, read_environment
from turnitover.repair.frozen_manifest import _file, load_frozen_manifest
from turnitover.repair.table1_judges import ObservationPair
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
    return metadata, loaded


def native_replay(manifest, native_root, output):
    """Cache only complete fresh native measurements bound to current source SHA."""
    cache = {}
    def replay(source):
        key = hashlib.sha256(source.encode()).hexdigest()
        if key not in cache:
            candidate = output / (key + '.ts')
            candidate.write_text(source)
            folder = output / ('native-' + key)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('rp29_native_replay.py')),
                '--native-root', str(native_root), '--manifest', str(manifest),
                '--candidate', str(candidate), '--output', str(folder)], capture_output=True, text=True,
                timeout=300)
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
             calibration, audit_factory=None, replay_factory=native_replay):
    output.mkdir(parents=True, exist_ok=False)
    replay = replay_factory(manifest, native_root, output)
    audit = (audit_factory(case, replay) if audit_factory else
             lambda source: {'passed': None, 'scope': 'unscored_trajectory',
                             'program_sha256': replay(source)['program_sha256'],
                             'reason': 'Independent full-defect checker unavailable'})
    def observe(source, folder):
        evidence = replay(source)
        return {'pairs': [ObservationPair(p['state'], Path(p['reference']), Path(p['candidate']))
                          for p in evidence['pairs']]}
    try:
        result = run_table1_repair(case['initial_program'].read_text(), generator, observe, audit,
            output / 'repair', condition=condition, max_rounds=rounds,
            reference_image=case['reference_image'], calibration=calibration,
            public_task={'description': case.get('public_task', ''),
                         'runtime_requirements': case.get('task_runtime_requirements', [])},
            allow_unscored=True)
    except Exception as exc:
        result_path = output / 'repair/result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            'status': 'incomplete', 'total_tokens': 0, 'error_type': type(exc).__name__}
    write(output / 'summary.json', result)
    return result


def execution_hashes(native_root):
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), Path(__file__).with_name('rp29_native_replay.py').resolve()]
    files += [root / name for name in (
        'turnitover/repair/table1_runner.py', 'turnitover/repair/table1_judges.py',
        'turnitover/repair/loop.py', 'turnitover/repair/patches.py',
        'turnitover/repair/frozen_manifest.py', 'turnitover/models/client.py',
        'turnitover/models/config.py', 'turnitover/verifier/contracts.py')]
    for folder in ('turnitover/core', 'turnitover/render', 'web/dist', 'configs'):
        files += [p for p in (native_root/folder).rglob('*')
                  if p.is_file() and p.suffix in {'.py', '.js', '.html', '.yaml'}]
    return {str(p): digest(p) for p in sorted(set(files))}


def _run(manifest, native_root, output, env, *, rounds=3, conditions=('none', 'pixel'),
        audit_factory=None, case_runner=run_case, collect_unscored=False, preflight_only=False):
    manifest, native_root, output = [Path(p).resolve() for p in (manifest, native_root, output)]
    metadata, cases = load_cases(manifest)
    if (not conditions or len(set(conditions)) != len(conditions)
            or set(conditions) - {'none', 'pixel'} or type(rounds) is not int or rounds < 1):
        raise ValueError('Unique none/pixel conditions and positive rounds required')
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
                'scoring': 'unscored_trajectory', 'success_rate': None,
                'resolved_rate': None, 'cost_to_success': None}
    if audit_factory is None and not collect_unscored:
        raise ValueError('Full independent scoring unavailable; --collect-unscored is required '
                         'to explicitly collect trajectories without Success/Resolved scores.')
    config = model_config('generator', env)
    config.validate()
    run_config = {'benchmark': metadata['benchmark'], 'manifest_sha256': digest(manifest),
                  'code_sha256': execution_hashes(native_root),
                  'endpoint_sha256': hashlib.sha256(config.base_url.encode()).hexdigest(),
                  'native_root': str(native_root), 'conditions': list(conditions), 'rounds': rounds,
                  'generator': config.public(), 'seed': None, 'seed_support': 'not_requested_or_verified',
                  'repetitions': 1, 'calibration_sha256': digest(calibration_path) if calibration else None,
                  'scoring': 'unscored_trajectory' if audit_factory is None else 'independent_callback'}
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
                    result = case_runner(path, case, native_root, folder,
                        lambda prompt, images: complete(config, prompt, images), condition, rounds,
                        calibration, audit_factory)
                except Exception as exc:
                    folder.mkdir(parents=True, exist_ok=True)
                    result_path = folder / 'repair/result.json'
                    result = json.loads(result_path.read_text()) if result_path.exists() else {
                        'status': 'incomplete', 'error_type': type(exc).__name__}
                    write(existing, result)
            rows.append({'id': case['id'], 'condition': condition, **result})
            write(output / 'progress.json', {'target': 109 * len(conditions), 'processed': len(rows),
                                             'complete': sum(r['status'] == 'complete' for r in rows)})
    summary = {'status': 'complete' if all(r['status'] == 'complete' for r in rows) else 'incomplete',
               'target': 109 * len(conditions), 'processed': len(rows), 'scoring': run_config['scoring'],
               'total_tokens': sum(r.get('total_tokens', 0) for r in rows),
               'success_rate': None, 'resolved_rate': None, 'cost_to_success': None,
               'results': rows, 'limitations': ['Missing independent scores are null, never pass or failure.',
                   'No automatic retry of interrupted or incomplete cases; costs may be incomplete after transport failures.']}
    write(output / 'summary.json', summary)
    return summary


def run(manifest, native_root, output, env, **kwargs):
    # A host-local lock prevents two queues spending tokens on the same output.
    key = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / ('rp29-batch-' + key + '.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process owns this batch output') from None
        return _run(manifest, native_root, output, env, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'native-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--preflight', action='store_true', help='Validate all frozen inputs without API calls')
    parser.add_argument('--collect-unscored', action='store_true', help='Explicitly allow incomplete independent scores')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--conditions', nargs='+', choices=('none', 'pixel'), default=['none', 'pixel'])
    args = parser.parse_args()
    if not args.preflight and args.env_file is None:
        parser.error('--env-file is required for execution')
    summary = run(args.manifest, args.native_root, args.output,
                  read_environment(args.env_file) if args.env_file else {},
                  rounds=args.rounds, conditions=args.conditions,
                  collect_unscored=args.collect_unscored, preflight_only=args.preflight)
    print(json.dumps(summary if args.preflight else {key: summary[key] for key in
                     ('status', 'target', 'processed', 'scoring', 'total_tokens')}, indent=2))
