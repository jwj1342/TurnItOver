"""Freeze a full, anonymous private review pack; never infer human approvals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image

ASSETS = Path(__file__).with_name('tier_review_assets')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(intake: Path, output: Path, custody: Path, namespace: str) -> dict:
    rows = json.loads(intake.read_text())
    if not isinstance(rows, list) or not rows or not namespace.strip():
        raise ValueError('A nonempty intake and explicit review namespace are required')
    if output.exists() or custody.exists() or custody.is_relative_to(output):
        raise ValueError('Pack and private custody outputs must be new and separate')
    seen, cases, mapping = set(), [], []
    output.mkdir(parents=True)
    custody.mkdir(parents=True)
    try:
        for row in rows:
            source_id = row['source_case_id']
            identity = str(Path(row['bundle']).resolve()) + ':' + source_id
            if identity in seen:
                raise ValueError('Duplicate intake case')
            seen.add(identity)
            cid = 'tr-' + hashlib.sha256((namespace + ':' + identity).encode()).hexdigest()[:20]
            case = {'case_id': cid, 'primary': [], 'supplementary': []}
            media = []
            for group in ('primary', 'supplementary'):
                for index, item in enumerate(row.get(group, [])):
                    source = Path(item['path']).resolve(strict=True)
                    if source.suffix.lower() != '.png':
                        raise ValueError('Only PNG media are accepted')
                    with Image.open(source) as img:
                        if group == 'primary' and img.size != (256, 256):
                            raise ValueError('Primary evidence must be native 256x256')
                    caption = item['caption']
                    if caption not in {f'{role} / {view} / {state}' for role in ('参考', '候选')
                                      for view in ('初始视角', '补充视角')
                                      for state in ('rest', 'mid', 'limit')} | {'三态辅助对比'}:
                        raise ValueError('Unexpected caption may leak source or answer')
                    name = f'media/{cid}/{group}-{index}.png'
                    target = output / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    case[group].append({'path': name, 'caption': caption})
                    media.append({'path': name, 'sha256': digest(target)})
            if not case['primary']:
                raise ValueError('Every case needs primary native evidence')
            cases.append(case)
            mapping.append({'case_id': cid, 'source_case_id': source_id,
                            'bundle': row['bundle'], 'media': media})
        cases.sort(key=lambda case: case['case_id'])
        encoded = json.dumps(cases, ensure_ascii=False, separators=(',', ':'))
        template = (ASSETS / 'review.template.html').read_text()
        (output / 'review.html').write_text(template.replace('{{CASES_JSON}}', encoded.replace('<', chr(92) + 'u003c')))
        for name in ('review.js', 'review.css', 'README.md'):
            shutil.copyfile(ASSETS / name, output / name)
        (output / 'case_manifest.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
        files = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()}
        package = {'version': 1, 'cases': len(cases), 'status': 'pending_human_review', 'files': files}
        (output / 'package.json').write_text(json.dumps(package, sort_keys=True, indent=2) + '\n')
        package_sha = digest(output / 'package.json')
        (output / 'package-sha.js').write_text('window.REVIEW_PACKAGE_SHA256=' + json.dumps(package_sha) + ';\n')
        (custody / 'mapping.json').write_text(json.dumps({'version': 1, 'package_sha256': package_sha,
                                                       'mapping': mapping}, ensure_ascii=False, indent=2) + '\n')
        (custody / 'intake.json').write_bytes(intake.read_bytes())
        return {'cases': len(cases), 'package_sha256': package_sha, 'status': 'pending_human_review'}
    except BaseException:
        shutil.rmtree(output)
        shutil.rmtree(custody)
        raise


def validate(root: Path) -> dict:
    package = json.loads((root / 'package.json').read_text())
    expected = set(package['files']) | {'package.json', 'package-sha.js'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Review pack inventory changed')
    for name, sha in package['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != sha:
            raise ValueError('Review pack file changed: ' + name)
    expected_js = 'window.REVIEW_PACKAGE_SHA256=' + json.dumps(digest(root / 'package.json')) + ';\n'
    if (root / 'package-sha.js').read_text() != expected_js:
        raise ValueError('Package hash binding changed')
    return {'cases': package['cases'], 'status': 'review_pack_verified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    for name in ('intake', 'output', 'custody'):
        build.add_argument('--' + name, type=Path, required=True)
    build.add_argument('--namespace', required=True)
    check = sub.add_parser('validate')
    check.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = (prepare(args.intake, args.output.resolve(), args.custody.resolve(), args.namespace)
              if args.command == 'build' else validate(args.root))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
