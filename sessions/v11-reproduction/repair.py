"""Portable entry points around the byte-preserved SOAR repair implementation.

Private inputs and outputs live in an explicitly supplied, initially empty workdir.
No command discovers credentials, connects to a server, or submits scheduler jobs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

HERE = Path(__file__).resolve().parent
CONDITIONS = ('none', 'pixel', 'ordinary', 'zero-shot-sweep', 'gold')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path, normalized=False):
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b'\r\n', b'\n') if normalized else raw).hexdigest()


def inside(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or out-of-root input: ' + str(name))
    return path


def check_sources(runtime=None):
    identity = read(HERE / 'provenance.json')
    for row in identity['frozen_files']:
        if sha(HERE / row['path']) != row['sha256']:
            raise ValueError('Frozen source changed: ' + row['path'])
    if runtime:
        for name, digest in identity['runtime_files_normalized_lf'].items():
            if sha(inside(runtime, name), normalized=True) != digest:
                raise ValueError('Runtime differs from pinned source: ' + name)
    return identity


def prepare(a):
    identity = check_sources(a.runtime)
    if a.out.exists():
        raise ValueError('Output must be a new directory; existing work is never overwritten')
    if sha(a.dataset / 'manifest.json') != identity['dataset_manifest_sha256']:
        raise ValueError('Requires the frozen v1.1 manifest')
    manifest = read(a.dataset / 'manifest.json')
    for name in ('annotations.private.jsonl', 'validation.episodes.jsonl', 'train.episodes.jsonl'):
        if sha(inside(a.dataset, name)) != manifest['files'][name]:
            raise ValueError('Dataset checksum mismatch: ' + name)
    train = [json.loads(line) for line in (a.dataset / 'train.episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(train) != 600:
        raise ValueError('Requires all 600 training cases for pixel calibration')
    for name in {name for row in train for name in row['images']}:
        if sha(inside(a.dataset, name)) != manifest['files'][name]:
            raise ValueError('Training image mismatch: ' + name)
    episodes = [json.loads(line) for line in (a.dataset / 'validation.episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    annotations = {r['id']: r for r in map(json.loads, (a.dataset / 'annotations.private.jsonl').read_text(encoding='utf-8').splitlines())}
    selected = [r for r in episodes if r['admission'] == 'eligible' and not r['is_clean']]
    ids = sorted(r['episode_id'] for r in selected)
    if len(ids) != 44 or hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest() != identity['case_ids_sha256']:
        raise ValueError('Requires the complete frozen 44-case set')
    cases = {c['case_id']: c for c in read(a.bundle / 'cases.json')}
    sums = {line.split(maxsplit=1)[1].lstrip('*'): line.split(maxsplit=1)[0]
            for line in (a.bundle / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()}
    files = set()
    rows = []
    for episode in selected:
        uid = episode['episode_id']
        case = cases[uid]
        annotation = annotations[uid]
        if annotation['split'] != 'validation' or annotation['admission'] != 'eligible' or annotation['is_clean']:
            raise ValueError('Episode and annotation admission disagree')
        if case['data_identity']['split_group'] != annotation['data_identity']['split_group']:
            raise ValueError('Source-object identity mismatch')
        for role in ('reference', 'candidate'):
            program = case['programs'][role]
            source = inside(a.bundle, program['typescript'])
            if sha(source) != program['typescript_sha256'] or sha(source)[:16] != annotation[role + '_sha']:
                raise ValueError('Program identity differs from v1.1')
            files.update((program['typescript'], program['spec']))
        for name in episode['images']:
            if sha(inside(a.dataset, name)) != manifest['files'][name]:
                raise ValueError('Frozen image mismatch: ' + name)
        rows.append(dict(case, annotation=annotation))
    if len({r['data_identity']['split_group'] for r in rows}) != 44:
        raise ValueError('Expected 44 independent source objects')
    for name in files:
        if sha(inside(a.bundle, name)) != sums.get(name):
            raise ValueError('Bundle checksum mismatch: ' + name)
    render_hashes = read(HERE / 'environment.json')['harness']
    for name, digest in render_hashes.items():
        if sha(inside(a.harness, name), normalized=True) != digest:
            raise ValueError('Renderer differs from reported experiment: ' + name)
    if not a.esbuild.is_file():
        raise ValueError('Supply the installed esbuild executable')
    a.out.mkdir(parents=True)
    (a.out / '.gitignore').write_text('*\n', encoding='utf-8')
    for name in files:
        target = a.out / 'assets-ready' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(a.bundle, name), target)
    shutil.copytree(HERE / 'frozen', a.out / 'code-v2', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write(a.out / 'selection.private.json', rows)
    inputs = {k: str(getattr(a, k).resolve()) for k in ('dataset', 'runtime', 'harness', 'esbuild')}
    inputs['selection_sha256'] = sha(a.out / 'selection.private.json')
    write(a.out / 'inputs.private.json', inputs)
    protocol = read(HERE / 'protocol-template.json')
    protocol['case_ids'] = ids
    write(a.out / 'protocol-v2.json', protocol)
    write(a.out / 'physical-scoring-protocol.json', read(HERE / 'physical-scoring-protocol.json'))
    write(a.out / 'trials.private.json', [{'case_id': r['case_id'], 'repetition': rep} for rep in range(1, 4) for r in rows])
    print('PREPARED', len(rows), 'cases; no model or renderer was executed')


def bootstrap(work):
    cfg = read(work / 'inputs.private.json')
    check_sources(Path(cfg['runtime']))
    if sha(work / 'selection.private.json') != cfg['selection_sha256']:
        raise ValueError('Prepared selection changed')
    manifest = read(Path(cfg['dataset']) / 'manifest.json')
    for name in ('validation.episodes.jsonl', 'annotations.private.jsonl', 'train.episodes.jsonl'):
        if sha(inside(Path(cfg['dataset']), name)) != manifest['files'][name]:
            raise ValueError('Dataset records changed: ' + name)
    for row in read(HERE / 'provenance.json')['frozen_files']:
        target = work / 'code-v2' / Path(row['path']).relative_to('frozen')
        if sha(target) != row['sha256']:
            raise ValueError('Prepared code changed; create a separate workspace')
    sys.path[:0] = [str(work / 'code-v2'), cfg['runtime']]
    from turnitover.repair import __path__ as repair_path
    repair_path.insert(0, str(work / 'code-v2/turnitover/repair'))
    import batch_common as common
    from dataclasses import replace

    def setup(root, batch):
        dataset = Path(cfg['dataset'])
        if sha(dataset / 'manifest.json') != common.MANIFEST:
            raise ValueError('Dataset manifest changed')
        records = {r['episode_id']: r for r in map(json.loads, (dataset / 'validation.episodes.jsonl').read_text(encoding='utf-8').splitlines())}
        cases = read(batch / 'selection.private.json')
        identity = read(HERE / 'provenance.json')
        ids = sorted(c['case_id'] for c in cases)
        if len(ids) != 44 or hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest() != identity['case_ids_sha256']:
            raise ValueError('Prepared case selection changed')
        for case in cases:
            record = records[case['case_id']]
            assert record['admission'] == 'eligible' and not record['is_clean']
            assert len(record['images']) == {'easy': 4, 'medium': 6, 'hard': 12}[case['difficulty']]
            for role in ('reference', 'candidate'):
                assert sha(inside(batch / 'assets-ready', case['programs'][role]['typescript'])) == case['programs'][role]['typescript_sha256']
        runtime = Path(cfg['runtime'])
        config = common.load_generate_config(runtime / 'configs/generate_replicacad.yaml', repo_root=runtime)
        config = replace(config, render=replace(config.render, harness_dir=Path(cfg['harness']), esbuild_bin=Path(cfg['esbuild'])))
        return dataset, records, cases, config

    common.setup = setup
    return common


def execute(work, name, args):
    sys.argv = [str(work / 'code-v2' / name)] + [str(x) for x in args]
    runpy.run_path(sys.argv[0], run_name='__main__')


def preflight(work):
    common = bootstrap(work)
    execute(work, 'calibrate_pixel.py', [read(work / 'inputs.private.json')['dataset'], work / 'pixel-calibration.json'])
    execute(work, 'preflight.py', [work, work, '--attempt', 'v4'])
    rows = read(work / 'preflight-v4/summary.json')
    if len(rows) != 44 or any(r['status'] != 'eligible' for r in rows):
        raise ValueError('Preflight did not pass for all 44 cases; no cases may be silently excluded')
    gold_preflight(work, common)
    protocol = read(work / 'protocol-v2.json')
    protocol.update(status='frozen', preflight_sha256=sha(work / 'preflight-v4/summary.json'), pixel_calibration_sha256=sha(work / 'pixel-calibration.json'))
    write(work / 'protocol-v2.json', protocol)


def gold_preflight(work, common):
    from types import SimpleNamespace
    from visible_gold_v4 import diagnose
    dataset, records, cases, cfg = common.setup(work, work)
    frames = {r['case_id']: r['framing'] for r in read(work / 'preflight-v4/summary.json')}
    ready, groups = [], []
    with common.ObservationSession(cfg.render, common.load_views(cfg.views_path)) as session:
        for case in cases:
            uid = case['case_id']
            ref, initial = common.sources(work, case)
            f = frames[uid]
            framing = common.Framing(tuple(f['center']), f['radius'])
            folder = work / 'gold-preflight-v2' / uid
            observe = common.observer(session, records[uid], case, common.ObjectProgram(ref), framing,
                                      lambda state, reference, candidate: SimpleNamespace(state=state, reference=reference, candidate=candidate))
            evidence = observe(initial, folder / 'initial')
            verdict = diagnose(initial, ref, session, case, framing, evidence)
            control = diagnose(ref, ref, session, case, framing, observe(ref, folder / 'reference'))
            if control['status'] != 'pass' or verdict['status'] != 'fail':
                raise ValueError('Gold initial/positive control failed; inspect, do not drop the case')
            hypotheses = read(folder / 'initial/gold-counterfactual-audit.private.json')['hypotheses']
            for label in case['annotation']['labels']:
                steps = {step for h in hypotheses if h['defect_id'] == label['defect_id'] and h['parts'] == label['parts'] for step in h['evidence_steps']}
                group = 'rest' if any(case['observation_cells'][step]['state'] == 'rest' for step in steps) else 'motion' if steps else 'unexposed'
                groups.append({'case_id': uid, 'defect_id': label['defect_id'], 'parts': label['parts'], 'group': group})
            ready.append({'case_id': uid, 'status': 'ready', 'initial_status': verdict['status'], 'reference_status': control['status']})
            write(work / 'gold-preflight-v2/summary.json', ready)
    counts = dict(collections.Counter(r['group'] for r in groups))
    if counts != {'rest': 9, 'motion': 42, 'unexposed': 3}:
        raise ValueError('Initial per-finding exposure differs from published groups: ' + str(counts))
    write(work / 'initial-defect-groups.json', groups)
    print('PREFLIGHT_OK', len(ready), counts)


def checked_trials(work, condition, trial_path):
    allowed = {(r['case_id'], r['repetition']) for r in read(work / 'trials.private.json')}
    requested = [(r['case_id'], r['repetition']) for r in read(trial_path)]
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= allowed:
        raise ValueError('Trial list must contain unique entries from the frozen 44 x 3 grid')
    for uid, rep in requested:
        folder = work / 'runs-v2' / condition / f'rep-{rep}' / uid
        result = folder / 'result.json'
        if folder.exists() and (not result.is_file() or read(result).get('status') != 'complete'):
            raise ValueError('Incomplete attempt preserved; archive and document an explicit retry before running')
    return requested


def run_worker(a):
    work = a.work
    bootstrap(work)
    protocol = read(work / 'protocol-v2.json')
    if protocol['status'] != 'frozen' or sha(work / 'preflight-v4/summary.json') != protocol['preflight_sha256']:
        raise ValueError('Run preflight first; its results must remain unchanged')
    if sha(work / 'pixel-calibration.json') != protocol['pixel_calibration_sha256']:
        raise ValueError('Pixel calibration changed')
    if os.environ.get('TIO_GENERATOR_MODEL') != 'qwen3.8-flash':
        raise ValueError('This delivery reproduces qwen3.8-flash only; a new generator needs a separately frozen protocol')
    for key in ('TIO_GENERATOR_BASE_URL', 'TIO_GENERATOR_API_KEY'):
        if not os.environ.get(key):
            raise ValueError('Missing environment variable ' + key)
    if a.condition == 'gold':
        groups = read(work / 'initial-defect-groups.json')
        if dict(collections.Counter(r['group'] for r in groups)) != {'rest': 9, 'motion': 42, 'unexposed': 3}:
            raise ValueError('Gold exposure preflight is incomplete')
    if a.condition == 'ordinary':
        ready = read(work / 'ordinary-queue-v2/ready.json')
        if ready.get('status') != 'ready' or ready.get('adapter') != 'ordinary-20261013':
            raise ValueError('Ordinary server is not ready')
    module = importlib.import_module('worker_gold_v4')
    if not os.environ.get('TIO_GENERATOR_RESOLVED_IP'):
        module.install_dns = lambda: None
    trial_path = a.trials or work / 'trials.private.json'
    requested = checked_trials(work, a.condition, trial_path)
    sys.argv = ['worker_gold_v4.py', str(work), str(work), a.condition, '--trial-list', str(trial_path)]
    module.main()
    failed = []
    for uid, rep in requested:
        p = work / 'runs-v2' / a.condition / f'rep-{rep}' / uid / 'result.json'
        if not p.exists() or read(p).get('status') != 'complete':
            failed.append((uid, rep))
    if failed:
        raise RuntimeError(f'{len(failed)} requested trials incomplete; preserved for explicit, versioned retry')


def main():
    if not __debug__:
        raise RuntimeError('Do not run with -O; frozen code uses assertions')
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('dataset', 'bundle', 'runtime', 'harness', 'esbuild', 'out'):
        prep.add_argument('--' + name, type=Path, required=True)
    for cmd in ('preflight', 'run', 'score', 'check'):
        parser = sub.add_parser(cmd)
        parser.add_argument('--work', type=Path, required=True)
        if cmd == 'run':
            parser.add_argument('--condition', choices=CONDITIONS, required=True)
            parser.add_argument('--trials', type=Path)
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve())
    if a.command == 'prepare':
        prepare(a)
    elif a.command == 'preflight':
        preflight(a.work)
    elif a.command == 'run':
        run_worker(a)
    elif a.command == 'score':
        bootstrap(a.work)
        execute(a.work, 'score_physical_batch_v4.py', [a.work, a.work])
    else:
        bootstrap(a.work)
        print('SOURCE_AND_LAYOUT_OK')


if __name__ == '__main__':
    main()
