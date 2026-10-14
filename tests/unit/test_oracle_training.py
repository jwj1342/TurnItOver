import hashlib
import json

import pytest

from scripts.evaluate_qwen_episode_sft import _status_metrics


def test_weighted_loss_excludes_prompt_and_matches_empirical_distribution():
    torch = pytest.importorskip('torch')
    from turnitover.oracle.training import weighted_assistant_loss
    logits = torch.tensor([[[9.,-9.],[2.,0.],[0.,2.]], [[-9.,9.],[2.,0.],[0.,2.]]], requires_grad=True)
    labels = torch.tensor([[-100,-100,0],[-100,-100,1]])
    loss = weighted_assistant_loss(logits, labels, [2/3,1/3])
    expected = -(2/3*torch.log_softmax(torch.tensor([2.,0.]),0)[0]+1/3*torch.log_softmax(torch.tensor([2.,0.]),0)[1])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.all(logits.grad[:,0]==0)
    with pytest.raises(ValueError):
        weighted_assistant_loss(logits,labels,[0.,1.])


def test_episode_loader_checks_formal_identity_images_and_terminal_filter(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    from turnitover.oracle.supervision import load_examples

    image_path = tmp_path / 'images/example.png'
    image_path.parent.mkdir()
    Image.new('RGB', (2, 2), 'red').save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    identity = {key: 'a' * 64 for key in ('source_identity_sha256', 'source_content_sha256')}
    identity.update(reference_sha='b' * 16, candidate_sha='c' * 16)
    identity['formal_training'] = True
    row = {'id': 'example:verdict', 'terminal': True, 'images': ['images/example.png'],
           'image_sha256': [digest], 'data_identity': identity,
           'messages': [{'role': 'user', 'content': 'input'},
                        {'role': 'assistant', 'content': '{"verdict": {}}'}]}
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'complete', 'formal_training': True}))
    for split in ('train', 'validation', 'test'):
        (tmp_path / f'{split}.episodes.jsonl').write_text(json.dumps(row) + '\n')

    loaded = list(load_examples(tmp_path, 'train', 'episodes',
                                require_formal_training=True, terminal_only=True))
    assert len(loaded) == 1 and loaded[0][1][0].size == (2, 2)

    row['image_sha256'] = ['0' * 64]
    (tmp_path / 'train.episodes.jsonl').write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='image SHA mismatch'):
        list(load_examples(tmp_path, 'train', 'episodes', require_formal_training=True))


def test_tokenize_example_masks_real_image_prompt_tokens():
    torch = pytest.importorskip('torch')
    Image = pytest.importorskip('PIL.Image')
    from turnitover.oracle.training import tokenize_example

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            length = 3 if len(messages) == 1 else 5
            return {'input_ids': torch.arange(length).reshape(1, -1),
                    'attention_mask': torch.ones((1, length), dtype=torch.long),
                    'pixel_values': torch.ones((1, 3, 2, 2))}

    record = {'images': ['image.png'], 'messages': [
        {'role': 'user', 'content': 'inspect'}, {'role': 'assistant', 'content': 'stop'}]}
    batch = tokenize_example(Processor(), record, [Image.new('RGB', (2, 2), 'blue')])
    assert batch['pixel_values'].shape == (1, 3, 2, 2)
    assert batch['labels'].tolist() == [[-100, -100, -100, 3, 4]]


def test_ordinary_and_evidence_training_views_keep_inputs_equal(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    from turnitover.oracle.sft_data import load_terminal_training_examples

    image_path = tmp_path / 'images/example.png'
    image_path.parent.mkdir()
    Image.new('RGB', (2, 2), 'red').save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    identity = {key: 'a' * 64 for key in ('source_identity_sha256', 'source_content_sha256')}
    identity.update(reference_sha='b' * 16, candidate_sha='c' * 16, formal_training=True,
                    asset_id='asset', source_dataset='fixture')
    context = {'task': 'inspect', 'parts': ['door'], 'joints': ['door'], 'views': ['front'],
               'runtime_properties': [], 'action_types': [], 'max_triangles': 1, 'max_draw_calls': 1}
    payload = {'context': context, 'history': [{'step': 0}], 'remaining': 0}
    evidence_target = {'verdict': {'status': 'uncertain', 'confidence': 0., 'findings': [],
                                   'summary': 'No visible evidence.', 'limitations': ['Incomplete.']}}
    row = {'id': 'episode:verdict', 'episode_id': 'episode', 'terminal': True,
           'condition': 'single_reference_episode_terminal', 'images': ['images/example.png'],
           'image_sha256': [digest], 'data_identity': identity,
           'messages': [{'role': 'user', 'content': 'protocol\n' + json.dumps(payload)},
                        {'role': 'assistant', 'content': json.dumps(evidence_target)}]}
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'complete', 'formal_training': True}))
    for split in ('train', 'validation', 'test'):
        (tmp_path / f'{split}.episodes.jsonl').write_text(json.dumps(row) + '\n')
    annotation = {'id': 'episode', 'split': 'train', 'reference_sha': 'b' * 16,
                  'candidate_sha': 'c' * 16, 'data_identity': {'source_dataset': 'fixture'},
                  'labels': [{'defect_id': 'kinematics.joint_axis', 'parts': ['door'],
                              'severity': .5, 'params': {'joint': 'door'}}]}
    (tmp_path / 'annotations.private.jsonl').write_text(json.dumps(annotation) + '\n')

    evidence = list(load_terminal_training_examples(tmp_path, 'train', 'evidence'))[0][0]
    ordinary = list(load_terminal_training_examples(tmp_path, 'train', 'ordinary'))[0][0]
    assert evidence['images'] == ordinary['images']
    assert evidence['messages'][0] == ordinary['messages'][0]
    assert json.loads(evidence['messages'][1]['content'])['verdict']['status'] == 'uncertain'
    target = json.loads(ordinary['messages'][1]['content'])['verdict']
    assert target['status'] == 'fail' and target['findings'][0]['evidence_steps'] == [0]


def test_multitask_training_adds_identical_stop_targets(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    from turnitover.oracle.sft_data import load_multitask_training_examples

    image_path = tmp_path / 'images/example.png'
    image_path.parent.mkdir()
    Image.new('RGB', (2, 2), 'red').save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    identity = {key: 'a' * 64 for key in ('source_identity_sha256', 'source_content_sha256')}
    identity.update(reference_sha='b' * 16, candidate_sha='c' * 16, formal_training=True,
                    asset_id='asset', source_dataset='fixture')
    context = {'task': 'inspect', 'parts': ['door'], 'joints': ['door'], 'views': ['front'],
               'runtime_properties': [], 'action_types': [], 'max_triangles': 1, 'max_draw_calls': 1}
    payload = {'context': context, 'history': [{'step': 0}], 'remaining': 0}
    target = {'verdict': {'status': 'uncertain', 'confidence': 0., 'findings': [],
                          'summary': 'No visible evidence.', 'limitations': ['Incomplete.']}}
    episode = {'id': 'episode:verdict', 'episode_id': 'episode', 'terminal': True,
               'condition': 'single_reference_episode_terminal', 'images': ['images/example.png'],
               'image_sha256': [digest], 'data_identity': identity,
               'messages': [{'role': 'user', 'content': 'protocol\n' + json.dumps(payload)},
                            {'role': 'assistant', 'content': json.dumps(target)}]}
    stop = {**episode, 'id': 'episode:stop:0', 'stop_target': 'stop',
            'stop_outcome': 'uncertain', 'stop_reason': 'budget_exhausted_uncertain',
            'messages': [episode['messages'][0], {'role': 'assistant', 'content': json.dumps({
                'stop_decision': {'decision': 'stop', 'outcome': 'uncertain', 'reason': 'done'}})}]}
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'complete', 'formal_training': True}))
    for split in ('train', 'validation', 'test'):
        (tmp_path / f'{split}.episodes.jsonl').write_text(json.dumps(episode) + '\n')
        (tmp_path / f'{split}.stopping.jsonl').write_text(json.dumps(stop) + '\n')
    annotation = {'id': 'episode', 'split': 'train', 'reference_sha': 'b' * 16,
                  'candidate_sha': 'c' * 16, 'data_identity': {'source_dataset': 'fixture'},
                  'labels': [{'defect_id': 'kinematics.joint_axis', 'parts': ['door'],
                              'severity': .5, 'params': {'joint': 'door'}}]}
    (tmp_path / 'annotations.private.jsonl').write_text(json.dumps(annotation) + '\n')

    ordinary = list(load_multitask_training_examples(tmp_path, 'train', 'ordinary'))
    evidence = list(load_multitask_training_examples(tmp_path, 'train', 'evidence'))
    assert [row['sft_objective'] for row, _ in ordinary] == ['diagnosis', 'stopping']
    assert ordinary[1][0]['messages'] == evidence[1][0]['messages']
    assert ordinary[0][0]['messages'][1] != evidence[0][0]['messages'][1]


def test_status_macro_f1_uses_only_supported_gold_classes():
    from collections import Counter
    confusion = {status: Counter() for status in ('fail', 'uncertain', 'pass')}
    confusion['fail'].update(fail=2, uncertain=1)
    confusion['uncertain'].update({'uncertain': 1, 'pass': 1})
    metrics = _status_metrics(confusion)
    assert metrics['labels'] == ['fail', 'uncertain']
    assert metrics['per_class']['fail']['f1'] == pytest.approx(0.8)
    assert metrics['per_class']['uncertain']['f1'] == pytest.approx(0.5)
    assert metrics['macro_f1'] == pytest.approx(0.65)
    assert metrics['confusion']['uncertain']['pass'] == 1
    assert sum(sum(metrics['confusion'][status].values())
               for status in ('uncertain', 'pass')) == 2
