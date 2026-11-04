import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.prepare_tier_review import prepare, validate


def test_review_pack_hides_identity_and_detects_changed_media(tmp_path):
    image = tmp_path / 'source.png'
    Image.new('RGB', (256, 256), 'white').save(image)
    intake = tmp_path / 'intake.json'
    intake.write_text(json.dumps([{'bundle': '/private/source', 'source_case_id': 'secret-case',
                                  'primary': [{'path': str(image), 'caption': '参考 / 初始视角 / rest'}]}]))
    pack, custody = tmp_path / 'pack', tmp_path / 'custody'
    assert prepare(intake, pack, custody, 'test-wave')['cases'] == 1
    assert validate(pack)['cases'] == 1
    assert 'secret-case' not in (pack / 'case_manifest.json').read_text()
    assert 'secret-case' in (custody / 'mapping.json').read_text()
    media = next((pack / 'media').rglob('*.png'))
    media.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        validate(pack)


def test_review_rejects_caption_with_answer(tmp_path):
    image = tmp_path / 'source.png'
    Image.new('RGB', (256, 256)).save(image)
    intake = tmp_path / 'intake.json'
    intake.write_text(json.dumps([{'bundle': 'x', 'source_case_id': 'case',
                                  'primary': [{'path': str(image), 'caption': 'hard defect'}]}]))
    with pytest.raises(ValueError, match='caption'):
        prepare(intake, tmp_path / 'pack', tmp_path / 'custody', 'wave')
    assert not (tmp_path / 'pack').exists()
