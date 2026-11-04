"""Build and check an offline Medium review package without source identities."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

if __package__:
    from .validate_easy_bundle import _png_size, _read_jsonl
    from .medium_bundle_inventory import validate_manifest, write_manifest
else:
    from validate_easy_bundle import _png_size, _read_jsonl
    from medium_bundle_inventory import validate_manifest, write_manifest

ASSETS = Path(__file__).with_name('medium_review_assets')
STATES = ['rest', 'mid', 'limit']
ROLES = ['reference', 'candidate']
STATIC = ['review.js', 'review.css', 'README.md']
BASE_FILES = {'case_manifest.json', 'blank_return.jsonl', 'review.html', *STATIC}
OPAQUE_ID = re.compile(r'mr-[0-9a-f]{16}')
SOURCE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}')
DECISIONS = {'medium', 'clean', 'too_obvious', 'too_subtle', 'invalid'}


def blank(case_id):
    return dict(case_id=case_id, reviewer='', decision=None, notes='', reviewed_at=None)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON field: {key}')
        result[key] = value
    return result


def _html(cases):
    # IDs and paths are constrained before this data reaches a script element.
    data = json.dumps(cases, ensure_ascii=False, separators=(',', ':'))
    return (ASSETS / 'review.template.html').read_text().replace('{{CASES_JSON}}', data)


def _validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError('review package needs at least one case')
    ids, media = [], set()
    for row in cases:
        if (not isinstance(row, dict) or set(row) != {'case_id', 'states', *ROLES}
                or not isinstance(row['case_id'], str) or not OPAQUE_ID.fullmatch(row['case_id'])
                or row['states'] != STATES):
            raise ValueError('review case schema includes unexpected identity or answer fields')
        cid = row['case_id']
        ids.append(cid)
        for role in ROLES:
            expected = {s: f'native/{cid}/{role}_{s}.png' for s in STATES}
            if row[role] != expected:
                raise ValueError('review frames must use only their opaque native paths')
            media.update(expected.values())
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError('review case IDs must be unique and in opaque order')
    return media


def validate_review_bundle(root: Path) -> dict:
    if Path(root).is_symlink():
        raise ValueError('review root must not be a symbolic link')
    root = Path(root).resolve()
    files = validate_manifest(root)
    encoded_cases = (root / 'case_manifest.json').read_text()
    cases = json.loads(encoded_cases, object_pairs_hook=_unique_object)
    if encoded_cases != json.dumps(cases, indent=2) + '\n':
        raise ValueError('review case manifest must use its canonical encoding')
    media = _validate_cases(cases)
    if files != BASE_FILES | media:
        raise ValueError('review package has unexpected or missing files')
    if (root / 'review.html').read_text() != _html(cases):
        raise ValueError('review HTML differs from its code-owned template')
    for name in STATIC:
        if (root / name).read_bytes() != (ASSETS / name).read_bytes():
            raise ValueError(f'review asset differs from code-owned template: {name}')
    expected_blank = [blank(row['case_id']) for row in cases]
    if (root / 'blank_return.jsonl').read_text() != ''.join(json.dumps(row) + '\n' for row in expected_blank):
        raise ValueError('review blank return contains answers or unexpected fields')
    if any(_png_size(root / name) != (256, 256) for name in media):
        raise ValueError('review media must remain native 256 by 256')
    return dict(cases=len(cases), media_files=len(media), blank_returns=len(expected_blank),
                manifest_files=len(files))


def prepare_review(root: Path, output: Path, namespace: str) -> list[dict]:
    if Path(root).is_symlink() or Path(output).is_symlink():
        raise ValueError('review source and output must not be symbolic links')
    root, output = Path(root).resolve(), Path(output).resolve()
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError('review namespace must be explicit and nonempty')
    if output.exists() or output.is_symlink() or output.is_relative_to(root):
        raise ValueError('review output must be new and outside the candidate bundle')
    rows = _read_jsonl(root / 'review_manifest.jsonl')
    ids = [row.get('case_id') for row in rows]
    if (not ids or any(not isinstance(cid, str) or not SOURCE_ID.fullmatch(cid) for cid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('source review IDs are invalid, duplicate, or empty')
    mapping = sorted([dict(case_id='mr-' + hashlib.sha256(f'{namespace}:{cid}'.encode()).hexdigest()[:16],
                           source_case_id=cid) for cid in ids], key=lambda r: r['case_id'])
    cases, copies = [], []
    for row in mapping:
        case = dict(case_id=row['case_id'], states=STATES.copy())
        for role in ROLES:
            case[role] = {}
            for state in STATES:
                source = root / f"native/{row['source_case_id']}/{role}_{state}.png"
                segments = source.relative_to(root).parts
                if (not source.is_file() or source.is_symlink()
                        or any(root.joinpath(*segments[:i]).is_symlink() for i in range(1, len(segments)))
                        or not source.resolve().is_relative_to(root) or _png_size(source) != (256, 256)):
                    raise ValueError(f'missing, symbolic or invalid native source frame: {source}')
                relative = f"native/{row['case_id']}/{role}_{state}.png"
                case[role][state] = relative
                copies.append((source, relative))
        cases.append(case)
    _validate_cases(cases)
    output.mkdir(parents=True)
    try:
        for source, relative in copies:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        (output / 'case_manifest.json').write_text(json.dumps(cases, indent=2) + '\n')
        (output / 'blank_return.jsonl').write_text(''.join(json.dumps(blank(c['case_id'])) + '\n' for c in cases))
        (output / 'review.html').write_text(_html(cases))
        for name in STATIC:
            shutil.copyfile(ASSETS / name, output / name)
        write_manifest(output)
        validate_review_bundle(output)
    except BaseException:
        shutil.rmtree(output)
        raise
    return mapping


def validate_return(root: Path, returned: Path) -> dict:
    """Check a real response without modifying it or assigning training eligibility."""
    validate_review_bundle(root)
    cases = json.loads((Path(root) / 'case_manifest.json').read_text())
    expected = {row['case_id'] for row in cases}
    rows = [json.loads(line, object_pairs_hook=_unique_object)
            for line in Path(returned).read_text().splitlines() if line.strip()]
    if (any(not isinstance(row, dict) or not isinstance(row.get('case_id'), str) for row in rows)
            or len(rows) != len(expected) or {row.get('case_id') for row in rows} != expected):
        raise ValueError('return must cover exactly the package cases without duplicates')
    reviewers, completed = set(), 0
    for row in rows:
        if (set(row) != set(blank(row['case_id'])) or not isinstance(row['reviewer'], str)
                or not isinstance(row['notes'], str)):
            raise ValueError('return contains unsupported fields or values')
        if row['reviewer'].strip():
            reviewers.add(row['reviewer'])
        if row['decision'] is None:
            if row['reviewed_at'] is not None:
                raise ValueError('blank decisions cannot carry a completion timestamp')
            continue
        if (not isinstance(row['decision'], str) or row['decision'] not in DECISIONS or not row['reviewer'].strip()
                or not isinstance(row['reviewed_at'], str)):
            raise ValueError('completed decision requires a reviewer and timestamp')
        try:
            stamp = datetime.fromisoformat(row['reviewed_at'].replace('Z', '+00:00'))
        except ValueError as error:
            raise ValueError('return timestamp must be ISO 8601') from error
        if stamp.utcoffset() is None:
            raise ValueError('return timestamp must include a timezone')
        completed += 1
    if len(reviewers) > 1:
        raise ValueError('one response file must belong to one reviewer')
    return dict(cases=len(rows), completed=completed, pending=len(rows) - completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    for name in ('root', 'output', 'mapping-output'):
        prepare.add_argument('--' + name, type=Path, required=True)
    prepare.add_argument('--namespace', required=True)
    validate = sub.add_parser('validate')
    validate.add_argument('--root', type=Path, required=True)
    response = sub.add_parser('validate-return')
    response.add_argument('--root', type=Path, required=True)
    response.add_argument('--return-file', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'validate':
        print(json.dumps(validate_review_bundle(args.root), indent=2))
    elif args.command == 'validate-return':
        print(json.dumps(validate_return(args.root, args.return_file), indent=2))
    else:
        if (args.mapping_output.exists() or args.mapping_output.is_symlink()
                or args.mapping_output.resolve().is_relative_to(args.output.resolve())):
            raise ValueError('private mapping must be new and outside the reviewer package')
        mapping = prepare_review(args.root, args.output, args.namespace)
        args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
        args.mapping_output.write_text(''.join(json.dumps(r) + '\n' for r in mapping))
        print(json.dumps(dict(cases=len(mapping), output=str(args.output))))


if __name__ == '__main__':
    main()
