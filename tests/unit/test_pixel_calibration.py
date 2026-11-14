import hashlib
import json

import pytest
from PIL import Image

from turnitover.repair.pixel_calibration import calibrate_pixel_threshold


def manifest(tmp_path):
    cases = []
    for index, (value, defective) in enumerate(((0, False), (128, True))):
        pairs = []
        for state in ('zero', 'mid', 'limit'):
            paths = {}
            hashes = {}
            for kind, intensity in (('reference', 0), ('candidate', value)):
                path = tmp_path / f'{index}-{state}-{kind}.png'
                Image.new('RGB', (2, 2), (intensity,) * 3).save(path)
                paths[kind] = path.name
                hashes[kind] = hashlib.sha256(path.read_bytes()).hexdigest()
            pairs.append({'state': state, **paths, 'sha256': hashes})
        cases.append({'id': str(index), 'defective': defective, 'pairs': pairs})
    path = tmp_path / 'train.json'
    path.write_text(json.dumps({'schema_version': 1, 'status': 'frozen', 'split': 'train', 'cases': cases}))
    return path


def test_selects_smallest_perfect_threshold(tmp_path):
    path = manifest(tmp_path)
    result = calibrate_pixel_threshold(path)
    assert result['threshold'] == 0
    assert result['balanced_accuracy'] == 1
    assert result['training_manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['tie_break'] == 'smallest_threshold'


def test_rejects_validation_and_changed_images(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data['split'] = 'validation'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Train'):
        calibrate_pixel_threshold(path)
    data['split'] = 'train'
    path.write_text(json.dumps(data))
    (tmp_path / data['cases'][0]['pairs'][0]['candidate']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        calibrate_pixel_threshold(path)


def test_requires_both_classes(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data['cases'] = data['cases'][:1]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='both clean and defective'):
        calibrate_pixel_threshold(path)
