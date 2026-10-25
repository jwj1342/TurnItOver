"""Synthetic fixtures exercise the audited export boundary without private data."""
import copy
import hashlib
import json

import pytest
from PIL import Image

from turnitover.oracle.episodes import episode_examples
from turnitover.oracle.sft_data import load_training_examples
from turnitover.oracle.supervision import load_examples


@pytest.fixture
def export(tmp_path):
    root = tmp_path
    Image.new('RGB', (4, 4), 'red').save(root / 'reference.png')
    Image.new('RGB', (4, 4), 'blue').save(root / 'observed.png')
    context = dict(task='inspect', parts=['door'], joints=['hinge'], views=['front'],
                   runtime_properties=[], action_types=['request_view'], max_triangles=1, max_draw_calls=1)
    labels = [dict(defect_id='structure.part_offset', parts=['door'], severity=.5, params={}),
              dict(defect_id='kinematics.joint_axis', parts=['door'], severity=.5, params={'joint': 'hinge'})]
    row = episode_examples('example', context, 'reference.png',
                           [dict(step=0, image_ref='observed.png', action={'type':'request_view', 'view_id':'front'})],
                           [0], [[0]], labels, 1)[-1]
    identity = dict(formal_training=True, source_dataset='fixture', source_asset_id='example',
                    asset_id='example', split_group='example', reference_sha='b'*16, candidate_sha='c'*16,
                    source_identity_sha256=hashlib.sha256(b'fixture:example').hexdigest(),
                    source_content_sha256='a'*64)
    row.update(data_identity=identity, image_sha256=[hashlib.sha256((root/p).read_bytes()).hexdigest()
                                                    for p in row['images']])
    annotation = dict(id='example', split='train', reference_sha='b'*16, candidate_sha='c'*16,
                      data_identity=identity, labels=labels)
    (root/'manifest.json').write_text(json.dumps(dict(status='complete', formal_training=True,
                                                     dataset_version='synthetic-v1')))
    (root/'annotations.private.jsonl').write_text(json.dumps(annotation)+'\n')
    (root/'train.episodes.jsonl').write_text(json.dumps(row)+'\n')
    return root, row


def write_row(root, row, split='train', kind='episodes'):
    (root/f'{split}.{kind}.jsonl').write_text(json.dumps(row)+'\n')


def test_equal_public_inputs_and_evidence_excludes_hidden_label(export):
    root, _ = export
    evidence = list(load_training_examples(root,'train','diagnosis','evidence'))[0][0]
    ordinary = list(load_training_examples(root,'train','diagnosis','ordinary'))[0][0]
    assert evidence['messages'][0] == ordinary['messages'][0]
    assert evidence['images'] == ordinary['images']
    assert len(json.loads(evidence['messages'][1]['content'])['verdict']['findings']) == 1
    assert len(json.loads(ordinary['messages'][1]['content'])['verdict']['findings']) == 2


@pytest.mark.parametrize('key', ['labels', 'future_history', 'private_target', 'ground_truth'])
def test_private_metadata_cannot_enter_public_observation(export, key):
    root, row = export
    payload = json.loads(row['messages'][0]['content'].rsplit('\n', 1)[-1])
    payload['history'][0]['payload'] = {key: ['hidden']}
    row['messages'][0]['content'] = 'protocol\n' + json.dumps(payload)
    write_row(root, row)
    with pytest.raises(ValueError, match='Private teacher metadata'):
        list(load_examples(root, 'train', 'episodes'))


def test_evidence_rejects_hidden_diagnosis_even_with_forged_condition(export):
    root, row = export
    ordinary = list(load_training_examples(root,'train','diagnosis','ordinary'))[0][0]
    row['condition'] = 'ordinary_final_diagnosis'
    row['messages'][1] = ordinary['messages'][1]
    write_row(root,row)
    with pytest.raises(ValueError, match='audited visible labels'):
        list(load_training_examples(root,'train','diagnosis','evidence'))


@pytest.mark.parametrize('mutation,match', [
    ('sha','image SHA mismatch'), ('absolute','relative'), ('traversal','outside dataset'),
    ('source','Source identity SHA mismatch'), ('public','image order')])
def test_reject_corrupted_exports(export, mutation, match):
    root, row = export
    if mutation == 'sha': row['image_sha256'][0] = '0'*64
    elif mutation == 'source': row['data_identity']['source_identity_sha256'] = '0'*64
    elif mutation == 'public':
        payload=json.loads(row['messages'][0]['content'].rsplit('\n',1)[-1]);payload['image_order']=[]
        row['messages'][0]['content']='protocol\n'+json.dumps(payload)
    else:
        row['images'][0] = str(root/'reference.png') if mutation == 'absolute' else '../reference.png'
        payload=json.loads(row['messages'][0]['content'].rsplit('\n',1)[-1]);payload['image_order']=row['images']
        row['messages'][0]['content']='protocol\n'+json.dumps(payload)
    write_row(root,row)
    with pytest.raises(ValueError, match=match): list(load_examples(root,'train','episodes'))


def test_reject_object_split_leakage_across_tasks(export):
    root,row=export
    other=copy.deepcopy(row);other['id']='other:stop'
    write_row(root,other,'validation','stopping')
    with pytest.raises(ValueError,match='leakage'): list(load_examples(root,'train','episodes'))


def test_reject_inconsistent_provenance(export):
    root,row=export
    other=copy.deepcopy(row);other['id']='other';other['episode_id']='other'
    other['data_identity']['source_content_sha256']='0'*64
    (root/'train.episodes.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(other)+'\n')
    with pytest.raises(ValueError,match='Inconsistent source'): list(load_examples(root,'train','episodes'))


def test_evidence_rejects_hidden_clean_pass(export):
    root,row=export
    target=json.loads(row['messages'][1]['content']);target['verdict'].update(status='pass',findings=[])
    row['messages'][1]['content']=json.dumps(target);write_row(root,row)
    with pytest.raises(ValueError,match='positive pass evidence'):
        list(load_training_examples(root,'train','diagnosis','evidence'))


def test_stopping_modes_have_identical_public_inputs_and_targets(export):
    root,row=export
    row['messages'][1]['content']=json.dumps({'stop_decision':{'decision':'stop','outcome':'fail','reason':'visible'}})
    write_row(root,row,kind='stopping')
    evidence=list(load_training_examples(root,'train','stopping','evidence'))[0][0]
    ordinary=list(load_training_examples(root,'train','stopping','ordinary'))[0][0]
    assert evidence == ordinary
    row['evidence']['covered_label_indices']=[];write_row(root,row,kind='stopping')
    with pytest.raises(ValueError,match='lacks audited visible evidence'):
        list(load_training_examples(root,'train','stopping','evidence'))


@pytest.mark.parametrize('indices', [[-1], [True], [9], [0,0]])
def test_stopping_rejects_invalid_audited_indices(export, indices):
    root,row=export
    row['messages'][1]['content']=json.dumps({'stop_decision':{'decision':'stop','outcome':'fail','reason':'visible'}})
    row['evidence']['covered_label_indices']=indices
    write_row(root,row,kind='stopping')
    with pytest.raises(ValueError,match='Invalid covered label indices'):
        list(load_training_examples(root,'train','stopping','evidence'))
