"""Synthetic return custody and classification tests; no real approval records."""
import json

import pytest
from PIL import Image

from scripts.import_tier_review import digest, import_return
from scripts.prepare_tier_review import prepare


def fixture(tmp_path, decision='hard'):
    image = tmp_path / 'image.png'
    Image.new('RGB', (256, 256), 'white').save(image)
    intake = tmp_path / 'intake.json'
    intake.write_text(json.dumps([{'source_case_id': 'source-case', 'bundle': str(tmp_path),
        'primary': [{'path': str(image), 'caption': '候选 / 初始视角 / rest'}]}]))
    pack, custody = tmp_path / 'pack', tmp_path / 'custody'
    result = prepare(intake, pack, custody, 'synthetic-import-test')
    mapping = custody / 'mapping.json'
    opaque = json.loads(mapping.read_text())['mapping'][0]['case_id']
    raw = tmp_path / 'actual-return.jsonl'
    raw.write_text(json.dumps({'case_id': opaque, 'reviewer': 'synthetic-reviewer',
        'decision': decision, 'notes': 'Synthetic motion observation' if decision else '',
        'reviewed_at': '2026-11-04' if decision else None,
        'review_package_sha256': result['package_sha256']}) + '\n')
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps([{'case_id': 'source-case', 'bundle': str(tmp_path),
                                    'kind': 'defect', 'tier': 'hard'}]))
    return raw, pack, mapping, metadata


@pytest.mark.parametrize('decision, expected', [('hard', 'approved'), ('clean', 'rejected'),
                                              ('easy', 'rejected'), (None, 'pending')])
def test_retains_raw_bytes_and_does_not_promote_mismatched_or_pending_judgments(tmp_path, decision, expected):
    raw, pack, mapping, metadata = fixture(tmp_path, decision)
    original = raw.read_bytes()
    output = tmp_path / 'imported'
    report = import_return(raw, digest(raw), pack, digest(pack / 'package.json'), mapping,
                           digest(mapping), metadata, digest(metadata), output)
    assert report['counts'] == {expected: 1}
    assert (output / 'raw-return.jsonl').read_bytes() == original
    returned = json.loads((output / 'return-0000.json').read_text())
    assert returned['raw_return']['sha256'] == digest(raw)
    assert returned['import_status'] == expected


@pytest.mark.parametrize('mutation', ['different_package', 'duplicate_opaque', 'missing_notes', 'raw_hash', 'mapping_media'])
def test_rejects_bad_actual_returns(tmp_path, mutation):
    raw, pack, mapping, metadata = fixture(tmp_path)
    original_sha = digest(raw)
    row = json.loads(raw.read_text())
    if mutation == 'different_package':
        row['review_package_sha256'] = 'f' * 64
    elif mutation == 'missing_notes':
        row['notes'] = ''
    elif mutation == 'raw_hash':
        row['notes'] = 'Changed raw return'
    elif mutation == 'mapping_media':
        mapping_document = json.loads(mapping.read_text())
        mapping_document['mapping'][0]['media'][0]['sha256'] = 'f' * 64
        mapping.write_text(json.dumps(mapping_document))
    raw.write_text(json.dumps(row) + '\n' + (json.dumps(row) + '\n' if mutation == 'duplicate_opaque' else ''))
    with pytest.raises(ValueError):
        import_return(raw, original_sha if mutation == 'raw_hash' else digest(raw), pack,
            digest(pack / 'package.json'), mapping, digest(mapping), metadata, digest(metadata), tmp_path / 'imported')
    assert not (tmp_path / 'imported').exists()
