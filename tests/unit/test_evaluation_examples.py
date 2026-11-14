"""Inference loader accepts clean targets and never opens other splits."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from tests.unit.test_observation_recognition import paired_record
from turnitover.oracle.evaluation_examples import load_evaluation_examples


def dataset(tmp_path):
    record = paired_record()
    payload = json.loads(record['messages'][0]['content'].rsplit('\n', 1)[-1])
    payload['context']['views'] = ['front']
    record['messages'][0]['content'] = 'Protocol\n' + json.dumps(payload)
    source = 'synthetic:asset'
    record.update(terminal=True, data_identity={
        'formal_training': True, 'source_dataset': 'synthetic', 'source_asset_id': 'asset',
        'asset_id': 'asset', 'split_group': source,
        'source_identity_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'source_content_sha256': 'c'*64, 'reference_sha': 'a'*16, 'candidate_sha': 'b'*16})
    verdict = dict(status='pass', confidence=1., findings=[], summary='Clean', limitations=[])
    record['messages'][1]['content'] = json.dumps({'verdict': verdict})
    files, hashes = {}, []
    for name in record['images']:
        Image.new('RGB', (2, 2)).save(tmp_path / name)
        digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        hashes.append(digest); files[name] = digest
    record['image_sha256'] = hashes
    data = (json.dumps(record)+'\n').encode()
    (tmp_path / 'validation.episodes.jsonl').write_bytes(data)
    files['validation.episodes.jsonl'] = hashlib.sha256(data).hexdigest()
    manifest = dict(status='complete', formal_training=True, files=files,
                    counts={'validation': 1}, groups={source: 'validation'})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return record, manifest


def test_only_validation_is_opened_and_clean_target_is_unchanged(tmp_path, monkeypatch):
    record, _ = dataset(tmp_path)
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert not path.name.startswith(('train.', 'test.'))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    examples = list(load_evaluation_examples(tmp_path))
    assert len(examples) == 1
    assert examples[0][0] == record
    assert len(examples[0][1]) == 6


@pytest.mark.parametrize('change', ['record', 'image', 'identity'])
def test_validation_rejects_unbound_or_invalid_inputs(tmp_path, change):
    record, manifest = dataset(tmp_path)
    if change == 'record':
        with (tmp_path / 'validation.episodes.jsonl').open('a') as stream:
            stream.write('\n')
    elif change == 'image':
        (tmp_path / record['images'][0]).write_bytes(b'changed')
    else:
        record['data_identity']['split_group'] = 'wrong'
        data = (json.dumps(record)+'\n').encode()
        (tmp_path / 'validation.episodes.jsonl').write_bytes(data)
        manifest['files']['validation.episodes.jsonl'] = hashlib.sha256(data).hexdigest()
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        list(load_evaluation_examples(tmp_path))


def test_test_split_is_rejected_before_opening_files(tmp_path):
    with pytest.raises(ValueError, match='Validation'):
        list(load_evaluation_examples(tmp_path, 'test'))
