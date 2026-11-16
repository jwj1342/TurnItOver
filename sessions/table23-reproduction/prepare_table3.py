"""Prepare a new private Table 3 workspace from locally available frozen inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
MANIFEST = 'b64e50f027a8788554f9248b6aa7da51b4010e16103f3ffafc9e6bb41806179b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or out-of-root input: {relative}')
    return path


def prepare(dataset, bundle, runtime, out):
    if out.exists():
        raise ValueError('Output must be a new directory; existing work is never overwritten')
    frozen = json.loads((HERE / 'provenance.json').read_text(encoding='utf-8'))
    if sha(dataset / 'manifest.json') != MANIFEST:
        raise ValueError('Requires the frozen v1.1 manifest')
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    # Validate source records against the file hashes bound by the manifest.
    checks = manifest['files']
    for name in ('annotations.private.jsonl', 'validation.episodes.jsonl'):
        if sha(dataset / name) != checks[name]:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    annotations = [json.loads(line) for line in (dataset / 'annotations.private.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = {a['id']: a for a in annotations if a['split'] == 'validation'
                and a['admission'] == 'eligible' and not a['is_clean'] and len(a['labels']) == 1}
    if len(selected) != 34:
        raise ValueError('Frozen protocol requires all 34 eligible single-defect cases')
    cases = {c['case_id']: c for c in json.loads((bundle / 'cases.json').read_text(encoding='utf-8'))}
    if not selected.keys() <= cases.keys():
        raise ValueError('Repair program bundle is missing selected cases')
    # The program bundle provides compiled programs and the exact rendering runtime.
    sums = {}
    for line in (bundle / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip('*')] = digest
    files = {'views.json', 'runtime/harness.js', 'runtime/three.esm.js', 'runtime/THREE-LICENSE.txt'}
    for uid, annotation in selected.items():
        case = cases[uid]
        if case['data_identity']['split_group'] != annotation['data_identity']['split_group']:
            raise ValueError('Source-object identity mismatch')
        for role in ('reference', 'candidate'):
            program = case['programs'][role]
            source = inside(bundle, program['typescript'])
            if sha(source) != program['typescript_sha256'] or sha(source)[:16] != annotation[role + '_sha']:
                raise ValueError('Program identity differs from v1.1')
            files.update((program['typescript'], program['javascript']))
    for name in files:
        if sha(inside(bundle, name)) != sums.get(name):
            raise ValueError(f'Program bundle checksum mismatch: {name}')
    for name, expected in frozen['table3_bundle_files'].items():
        if sha(inside(bundle, name)) != expected:
            raise ValueError(f'Render configuration differs from reported experiment: {name}')
    for name, expected in frozen['runtime_files'].items():
        normalized = inside(runtime, name).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(normalized).hexdigest() != expected:
            raise ValueError(f'Runtime source differs from pinned commit: {name}')
    out.mkdir(parents=True)
    (out / 'inputs').mkdir()
    shutil.copy2(dataset / 'manifest.json', out / 'inputs/manifest.json')
    for name in files:
        dest = out / 'prepared' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(bundle, name), dest)
    selection = [dict(cases[uid], annotation=selected[uid]) for uid in sorted(selected)]
    (out / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    target = out / 'source/three-tier-900-sft-v1.1-runtime/turnitover'
    shutil.copytree(runtime / 'turnitover', target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (HERE / 'table3').iterdir():
        if path.suffix in ('.py', '.html'):
            shutil.copy2(path, out / path.name)
    # Generated inputs, labels, programs, images, and browser profiles stay private.
    (out / '.gitignore').write_text('*\n', encoding='utf-8')
    print(f'Prepared {len(selection)} cases in {out}; no model or rendering was run')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'repair-bundle', 'runtime', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.dataset.resolve(), a.repair_bundle.resolve(), a.runtime.resolve(), a.out.resolve())


if __name__ == '__main__':
    main()
