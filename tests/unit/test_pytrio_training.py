"""Offline orchestration tests: no SDK/network or model weights are required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/train_pytrio_paired_sft.py'


def load_cli():
    spec = importlib.util.spec_from_file_location('train_pytrio_test_cli', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_run(tmp_path, monkeypatch, *, mode='evidence', loss=.5):
    import turnitover.oracle.paired_observations as pairs
    import turnitover.oracle.pytrio_adapter as adapter
    import turnitover.oracle.sft_data as data
    import turnitover.oracle.sft_metrics as metrics

    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    for filename, content in [('manifest.json', '{"dataset_version":"paired-test"}'),
                              ('train.episodes.jsonl', '{}\n'), ('annotations.private.jsonl', '{}\n')]:
        (dataset / filename).write_text(content)
    args = SimpleNamespace(dataset=dataset, out=tmp_path / 'output', supervision=mode,
                           dataset_manifest_sha256=hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest(),
                           epochs=3, learning_rate=2e-4, rank=8, alpha=None, model='test/model', seed=20261013)
    records = [{'id': name, 'messages': [{'role': 'user', 'content': 'pair input'},
                                        {'role': 'assistant', 'content': f'{mode}:{name}'}]}
               for name in ('a', 'b')]
    selections = []

    def examples(root, split, task, supervision):
        selections.append((root, split, task, supervision))
        return [(record, []) for record in records]

    monkeypatch.setattr(data, 'load_training_examples', examples)
    monkeypatch.setattr(data, 'supervision_digest', lambda records: 'supervision-hash')
    monkeypatch.setattr(metrics, 'public_input_digest', lambda records: 'public-hash')
    monkeypatch.setattr(pairs, 'is_paired', lambda record: True)
    calls = SimpleNamespace(encoded=[], forward=[], optim=[], saved=[], created=[], clients=0)
    weights = SimpleNamespace(to_numpy=lambda: np.ones(3))

    def encode(record, images, tokenizer, processor, **kwargs):
        calls.encoded.append((record['id'], kwargs['completion']))
        return SimpleNamespace(loss_fn_inputs={'weights': weights})

    monkeypatch.setattr(adapter, 'model_input', encode)

    def future(value):
        return SimpleNamespace(result=lambda: value)

    class Training:
        def forward_backward(self, datums, loss_fn):
            calls.forward.append((len(datums), loss_fn))
            return future(SimpleNamespace(metrics={'loss_mean': loss, 'token_count': 3}))

        def optim_step(self, params):
            calls.optim.append(params)
            return future(None)

        def save_weights_for_sampler(self, name):
            calls.saved.append(name)
            return future(SimpleNamespace(path='pytrio://saved-adapter'))

    class Client:
        def __init__(self):
            calls.clients += 1

        def create_lora_training_client(self, **kwargs):
            calls.created.append(kwargs)
            return Training()

        def create_sampling_client(self, **kwargs):
            return SimpleNamespace(get_tokenizer=lambda: SimpleNamespace(name_or_path='test/model'))

    sdk = SimpleNamespace(ServiceClient=Client, AdamParams=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, 'pytrio', sdk)
    processor = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoImageProcessor=processor))
    return load_cli(), args, calls, selections


@pytest.mark.parametrize('mode', ['ordinary', 'evidence'])
def test_epochs_and_supervision_are_honored_and_adapter_saved(tmp_path, monkeypatch, mode):
    cli, args, calls, selections = setup_run(tmp_path, monkeypatch, mode=mode)
    report = cli.train(args)
    assert selections == [(args.dataset, 'train', 'diagnosis', mode)]
    assert calls.created == [{'base_model': args.model, 'rank': 8, 'seed': args.seed}]
    assert len(calls.forward) == len(calls.optim) == 6
    assert all(item == (1, 'cross_entropy') for item in calls.forward)
    assert all(params == {'learning_rate': 2e-4} for params in calls.optim)
    for name in ('a', 'b'):
        assert calls.encoded.count((name, f'{mode}:{name}')) == 3
    assert calls.saved == [f'paired-sft-{mode}']
    assert report['status'] == 'complete' and report['steps'] == 6
    persisted = json.loads((args.out / 'run.json').read_text())
    assert persisted == report
    assert persisted['public_input_sha256'] == 'public-hash'
    assert len(persisted['target_sha256']) == 64
    assert persisted['lora']['alpha'] is None
    assert persisted['weights_path'] == 'pytrio://saved-adapter'
    rows = [json.loads(line) for line in (args.out / 'loss.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(6))
    assert [row['epoch'] for row in rows] == [0, 0, 1, 1, 2, 2]


def test_nonfinite_loss_records_failure_without_optimizer_or_save(tmp_path, monkeypatch):
    cli, args, calls, _ = setup_run(tmp_path, monkeypatch, loss=float('nan'))
    with pytest.raises(ValueError, match='non-finite'):
        cli.train(args)
    report = json.loads((args.out / 'run.json').read_text())
    assert report['status'] == 'failed' and report['steps'] == 0
    assert report['error_type'] == 'ValueError'
    assert not calls.optim and not calls.saved


def test_alpha_rejected_before_import_or_service_creation(tmp_path, monkeypatch):
    cli, args, calls, _ = setup_run(tmp_path, monkeypatch)
    args.alpha = 16
    # Even a deliberately unusable SDK must never be accessed for this request.
    monkeypatch.setitem(sys.modules, 'pytrio', None)
    with pytest.raises(ValueError, match='does not expose alpha'):
        cli.train(args)
    assert calls.clients == 0 and not args.out.exists()
