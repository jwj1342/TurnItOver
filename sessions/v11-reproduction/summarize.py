"""Collect a local private run tree, or independently reaggregate an existing archive."""
import argparse
import datetime
from pathlib import Path
import runpy
import shutil
from repair import CONDITIONS, HERE, read, write, sha


def collect(work, out):
    data = {}
    measurements = []
    for condition in CONDITIONS:
        rows = []
        for path in sorted((work / 'runs-v2' / condition).glob('rep-*/*/result.json')):
            result = read(path)
            folder = path.parent
            row = {'case_id': folder.name, 'repetition': folder.parent.name,
                   **{k: result.get(k) for k in ('status', 'termination', 'final_passed', 'completed_rounds', 'first_success_round', 'generator_calls', 'judge_calls', 'total_tokens', 'judge_tokens', 'usage_complete', 'error_type')}}
            rows.append(row)
            if result.get('status') != 'complete':
                continue
            measurement = {'condition': condition, 'repetition': int(folder.parent.name.split('-')[1]), 'case_id': folder.name,
                           'result_sha256': sha(path), 'source_sha256': sha(folder / 'final.ts'),
                           'result': {k: v for k, v in result.items() if k not in ('rounds', 'final_audit', 'audit_roster')}, 'rounds': []}
            if result['final_passed'] != result['final_audit']['passed']:
                raise ValueError('Final success differs from independent audit')
            for p in [folder] + sorted(folder.glob('round-*')):
                if not p.is_dir():
                    continue
                audit = read(p / 'audit.private.json') if (p / 'audit.private.json').exists() else {}
                verdict = read(p / 'judge.json') if (p / 'judge.json').exists() else {}
                invalid = read(p / 'invalid-verdict.json') if (p / 'invalid-verdict.json').exists() else None
                measurement['rounds'].append({'round': 0 if p == folder else int(p.name.split('-')[1]), 'usable': audit.get('usable'),
                                              'passed': audit.get('passed'), 'verdict': verdict.get('status'),
                                              'findings': verdict.get('feedback', {}).get('findings', []), 'invalid': invalid,
                                              'accepted_source_exists': (p / 'accepted.ts').exists()})
            measurements.append(measurement)
        data[condition] = {'results': rows}
    write(out / 'monitor-latest.json', {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'data': data})
    write(out / 'measurements.json', measurements)
    shutil.copyfile(work / 'selection.private.json', out / 'selection.private.json')
    shutil.copyfile(work / 'physical-scores-v1/all-scores.json', out / 'latest-physical-scores-v1-all-scores.json')


def summarize(archive, out, work=None):
    if out.exists():
        raise ValueError('Output must be a new directory')
    out.mkdir(parents=True)
    (out / '.gitignore').write_text('*\n', encoding='utf-8')
    if work:
        collect(work, out)
    else:
        for name in ('monitor-latest.json', 'measurements.json', 'selection.private.json', 'latest-physical-scores-v1-all-scores.json'):
            shutil.copyfile(archive / name, out / name)
    selection = read(out / 'selection.private.json')
    ids = {r['case_id'] for r in selection}
    if len(ids) != 44 or len({r['data_identity']['split_group'] for r in selection}) != 44:
        raise ValueError('Requires 44 cases and source objects')
    identity = read(HERE / 'provenance.json')
    import hashlib
    if hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode()).hexdigest() != identity['case_ids_sha256']:
        raise ValueError('Archive case identity differs from the frozen set')
    snapshot = read(out / 'monitor-latest.json')
    for filename in ('measurements.json', 'latest-physical-scores-v1-all-scores.json'):
        records = read(out / filename)
        keys = [(r['condition'], r['case_id'], r['repetition']) for r in records]
        if len(keys) != len(set(keys)) or any(c not in CONDITIONS or uid not in ids or rep not in (1, 2, 3) for c, uid, rep in keys):
            raise ValueError('Duplicate or out-of-protocol measurements: ' + filename)
    for condition in CONDITIONS:
        rows = snapshot['data'][condition]['results']
        keys = [(r['case_id'], r['repetition']) for r in rows]
        if len(keys) != len(set(keys)) or any(uid not in ids or rep not in ('rep-1', 'rep-2', 'rep-3') for uid, rep in keys):
            raise ValueError('Duplicate or out-of-protocol trajectory')
    # Exact original aggregation code; ROOT resolves to this new private output directory.
    script = out / 'analyze_batch.py'
    shutil.copyfile(HERE / 'frozen/analyze_batch.py', script)
    runpy.run_path(str(script), run_name='__main__')
    report = read(out / 'analysis.json')
    for name, row in report['conditions'].items():
        print(name, row['n'], row['success'], row.get('source_object_bootstrap_ci95'))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--archive', type=Path)
    sources.add_argument('--work', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    summarize(a.archive, a.out.resolve(), a.work)
