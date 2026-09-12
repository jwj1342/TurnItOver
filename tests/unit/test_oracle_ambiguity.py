import json

import pytest

from turnitover.oracle.ambiguity import export_distributions
from turnitover.oracle.supervision import action_examples


def test_exact_public_prefix_groups_paths_and_keeps_splits_separate(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'manifest.json').write_text('{"status":"complete"}')
    context = dict(task='verify', parts=['door'], joints=['hinge'], views=['front','back'],
                   runtime_properties=[], action_types=['request_view'], max_triangles=5000, max_draw_calls=32)
    rows=[]
    for index, view in enumerate(('front','front','back')):
        reference=f'{index}.png'
        (source / reference).write_bytes(b'identical image bytes')
        history=[dict(step=0, action=dict(type='request_view', view_id=view), image_ref='future.png', payload={})]
        rows.extend(action_examples(str(index), context, reference, history, 1))
    (source / 'train.actions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    # Same visible input in a different split must not add to train frequencies.
    (source / 'validation.actions.jsonl').write_text(json.dumps(rows[2]))
    (source / 'test.actions.jsonl').write_text('')
    output=tmp_path / 'out'
    report=export_distributions(source,output)
    assert report['conflicting_groups']==1
    assert report['affected_source_examples']==3
    train=[json.loads(s) for s in (output/'train.actions.jsonl').read_text().splitlines()]
    weights={json.loads(r['messages'][1]['content'])['action']['view_id']:r['loss_weight'] for r in train}
    assert weights == pytest.approx(dict(front=2/3,back=1/3))
    validation=json.loads((output/'validation.actions.jsonl').read_text())
    assert validation['loss_weight']==1.
    assert 'members' not in train[0]
    for r in train:
        assert 'future.png' not in json.dumps(r)
        for relative in r['images']:
            assert (output/relative).read_bytes()==b'identical image bytes'
