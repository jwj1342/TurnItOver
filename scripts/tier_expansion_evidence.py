"""Verify local evidence bindings; never create or infer human approvals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

if __package__:
    from .prepare_tier_review import validate as validate_review_pack
    from .tier_review_records import classify, verify_sets
else:
    from prepare_tier_review import validate as validate_review_pack
    from tier_review_records import classify, verify_sets

SHA256 = re.compile(r'[0-9a-f]{64}')


def _file(root: Path, reference: dict) -> Path:
    if not isinstance(reference, dict):
        raise ValueError('Evidence requires a path and SHA256')
    name, digest = reference.get('path'), reference.get('sha256')
    if not isinstance(name, str) or not name or not SHA256.fullmatch(str(digest)):
        raise ValueError('Evidence requires a path and SHA256')
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Evidence paths must stay within the private manifest directory')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or escaping evidence file: {name}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Evidence SHA256 differs: {name}')
    return path


def _document(root: Path, reference: dict) -> dict:
    value = json.loads(_file(root, reference).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('Evidence document must be a version 1 object')
    return value


def validate_case_evidence(root: Path, row: dict, case: dict) -> None:
    """Bind actual reports, reviewed media and returned judgment to this case.

    A file and reviewer field do not authenticate human identity. Review custody
    remains an external responsibility. No approvals are generated here.
    """
    evidence = case.get('evidence')
    if not isinstance(evidence, dict):
        raise ValueError(f"Added case lacks local evidence references: {case['case_id']}")
    automatic = _document(root, evidence.get('automatic_report'))
    package = _document(root, evidence.get('review_package'))
    returned = _document(root, evidence.get('review_return'))
    binding = {'case_id': case['case_id'], 'source_identity': row['source_identity'],
               'geometry_fingerprint': row['geometry_fingerprint'], 'tier': row['tier'],
               'kind': case['kind'], 'defect_id': case.get('defect_id')}
    if automatic.get('binding') != binding or automatic.get('status') != 'passed':
        raise ValueError('Automatic evidence case/source/geometry binding or status differs')
    media = automatic.get('media')
    if not isinstance(media, list) or not media:
        raise ValueError('Automatic evidence requires nonempty media inventory')
    paths = [_file(root, reference) for reference in media]
    if len(set(paths)) != len(paths):
        raise ValueError('Automatic media inventory has duplicate paths')
    if case.get('automatic_evidence_sha256') != evidence['automatic_report']['sha256']:
        raise ValueError('Automatic evidence reference differs from admission hash')
    if package.get('case_id') != case['case_id'] or package.get('media') != media:
        raise ValueError('Review package differs from the checked case media')
    if returned.get('case_id') != case['case_id']:
        raise ValueError('Human return belongs to a different case')
    if returned.get('review_package_sha256') != package.get('blind_package_sha256'):
        raise ValueError('Human return is not bound to the reviewed package')
    if returned.get('difficulty_review') != case['difficulty_review']:
        raise ValueError('Human return judgment differs from admission record')
    returned_root = _file(root, evidence['review_return']).parent
    raw_path = _file(returned_root, returned.get('raw_return'))
    mapping = _document(returned_root, returned.get('custody_mapping'))
    metadata_path = _file(returned_root, returned.get('case_metadata'))
    blind_path = _file(returned_root, returned.get('blind_package'))
    blind_sha = returned['blind_package']['sha256']
    if blind_sha != returned['review_package_sha256'] or mapping.get('package_sha256') != blind_sha:
        raise ValueError('Raw return custody package binding differs')
    validate_review_pack(blind_path.parent)
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    blind_cases = json.loads((blind_path.parent / 'case_manifest.json').read_text())
    verify_sets(raw_rows, mapping['mapping'], blind_cases, blind_sha)
    selected = [item for item in mapping['mapping'] if item['case_id'] == returned.get('opaque_case_id')
                and item['source_case_id'] == case['case_id']]
    if len(selected) != 1:
        raise ValueError('Opaque review mapping differs from admitted case')
    mapped_media = selected[0]['media']
    if sorted(item['sha256'] for item in mapped_media) != sorted(item['sha256'] for item in media):
        raise ValueError('Blind reviewer saw different media from automatic evidence')
    pack_files = json.loads(blind_path.read_text())['files']
    if any(pack_files.get(item['path']) != item['sha256'] for item in mapped_media):
        raise ValueError('Custody mapping media differs from frozen review pack')
    metadata = [item for item in json.loads(metadata_path.read_text()) if item['case_id'] == case['case_id']]
    if len(metadata) != 1 or metadata[0]['tier'] != row['tier'] or metadata[0]['kind'] != case['kind']:
        raise ValueError('Returned review expected labels differ from admission')
    if str(Path(metadata[0]['bundle']).resolve()) != str(Path(selected[0]['bundle']).resolve()):
        raise ValueError('Returned review source bundle differs from custody')
    raw_case = next(item for item in raw_rows if item['case_id'] == returned['opaque_case_id'])
    status, judgment = classify(raw_case, metadata[0])
    if status != 'approved' or returned.get('import_status') != status or judgment != case['difficulty_review']:
        raise ValueError('Original human judgment does not approve this case')
