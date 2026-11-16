import copy
import hashlib
import json

import pytest

from tests.unit.test_sft_data import export, write_row
from turnitover.oracle.clean_pass import POLICY, clean_verdict, observation_digest
from turnitover.oracle.sft_data import load_training_examples, validate_target


@pytest.fixture
def certified(export):
    root, row = export
    annotation = json.loads((root/'annotations.private.jsonl').read_text())
    annotation['labels'] = []
    annotation['candidate_sha'] = annotation['reference_sha']
    row['data_identity']['candidate_sha'] = row['data_identity']['reference_sha']
    annotation['data_identity'] = copy.deepcopy(row['data_identity'])
    row['evidence']['covered_label_indices'] = []
    row['evidence']['uncovered_label_indices'] = []
    for x in (row, annotation): x.update(label_policy=POLICY,is_clean=True,admission='eligible')
    cert = dict(policy=POLICY,basis='published_reference_identity_and_complete_frozen_observation',
                scope='published_structure_kinematics_fixed_observation_task',case_id='example',
                reference_sha=annotation['reference_sha'],observation_sha256=observation_digest(row))
    row['pass_certificate'] = cert
    annotation['pass_certificate'] = copy.deepcopy(cert)
    row['messages'][1]['content'] = json.dumps(clean_verdict())
    return root,row,annotation


def persist(root,row,annotation):
    write_row(root,row)
    (root/'annotations.private.jsonl').write_text(json.dumps(annotation)+'\n')
    manifest=json.loads((root/'manifest.json').read_text())
    manifest.update(label_policy=POLICY,files={p:hashlib.sha256((root/p).read_bytes()).hexdigest()
        for p in ['train.episodes.jsonl','annotations.private.jsonl']})
    (root/'manifest.json').write_text(json.dumps(manifest))


def test_certified_pass_loads_equally_for_both_conditions(certified):
    root,row,a=certified;persist(root,row,a)
    evidence=list(load_training_examples(root,'train','diagnosis','evidence'))[0][0]
    ordinary=list(load_training_examples(root,'train','diagnosis','ordinary'))[0][0]
    assert evidence['messages']==ordinary['messages']
    assert json.loads(evidence['messages'][1]['content'])['verdict']['status']=='pass'


@pytest.mark.parametrize('mutation',['missing_certificate','changed_image','shortened_history','not_terminal','changed_program','wrong_case','hidden_defect'])
def test_pass_requires_correct_case_program_and_complete_observation(certified,mutation):
    _,row,a=certified
    if mutation=='missing_certificate': row.pop('pass_certificate')
    elif mutation=='changed_image': row['image_sha256'][0]='0'*64
    elif mutation=='shortened_history':
        payload=json.loads(row['messages'][0]['content'].rsplit('\n',1)[-1]);payload['remaining']=1
        row['messages'][0]['content']='protocol\n'+json.dumps(payload)
    elif mutation=='not_terminal': row['terminal']=False
    elif mutation=='changed_program': row['data_identity']['candidate_sha']='a'*16
    elif mutation=='wrong_case': row['episode_id']='other'
    elif mutation=='hidden_defect': a['labels']=[{'defect_id':'structure.part_offset','parts':['door']}]
    with pytest.raises(ValueError): validate_target(row,'diagnosis','evidence',a)


def test_stopping_pass_uses_same_certificate(certified):
    _,row,a=certified
    row['messages'][1]['content']=json.dumps({'stop_decision':dict(decision='stop',outcome='pass',reason='scope verified')})
    validate_target(row,'stopping','evidence',a)
    row.pop('pass_certificate')
    with pytest.raises(ValueError):validate_target(row,'stopping','evidence',a)


def test_quarantine_excluded_without_erasing_raw_rows(certified):
    root,row,a=certified
    row['admission']=a['admission']='quarantined_pending_adjudication'
    persist(root,row,a)
    for mode in ('ordinary','evidence'):
        assert list(load_training_examples(root,'train','diagnosis',mode))==[]
    assert len((root/'train.episodes.jsonl').read_text().splitlines())==1


def test_metadata_edit_invalidates_release_hash(certified):
    root,row,a=certified;persist(root,row,a)
    row['is_clean']=False;write_row(root,row)
    with pytest.raises(ValueError,match='metadata SHA'):
        list(load_training_examples(root,'train','diagnosis','evidence'))


def test_private_certificate_cannot_leak_into_prompt(certified):
    root,row,a=certified
    payload=json.loads(row['messages'][0]['content'].rsplit('\n',1)[-1])
    payload['history'][0]['payload']={'pass_certificate':row['pass_certificate']}
    row['messages'][0]['content']='protocol\n'+json.dumps(payload)
    persist(root,row,a)
    with pytest.raises(ValueError,match='Private teacher metadata'):
        list(load_training_examples(root,'train','diagnosis','evidence'))
