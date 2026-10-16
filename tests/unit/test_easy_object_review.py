"""Artifact-level split, isolation and blind-review contract checks."""
import json

from scripts.prepare_easy_object_review import DEFAULT_OUT, gate
from scripts.prepare_obvious_defect_review import DEFAULT_SPLITS, sha256_path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_easy_bundle_is_train_only_blind_and_normal_first():
    root = DEFAULT_OUT
    manifest = json.loads((root/'manifest.json').read_text())
    for file, sha in manifest['files'].items():
        assert sha256_path(root/file) == sha
    selected = [r for r in rows(root/'private/object_selection.jsonl') if r['selected']]
    summary = json.loads((root/'private/summary.json').read_text())
    assert sha256_path(root/'private/object_selection.jsonl') == summary['selection_sha256_before_injection']
    assert len(selected) == 8
    splits = json.loads(DEFAULT_SPLITS.read_text())['groups']
    assert all(splits['PartNet-Mobility:'+r['asset_id'][3:]] == 'train' for r in selected)
    assert not ({r['asset_id'] for r in selected} & set(summary['easy_objects']['baseline_assets_excluded']))
    public = rows(root/'review_manifest.jsonl')
    assert len(public) == 24 and len({r['board'] for r in public}) == 8
    assert all(not {'kind','defect_id','label','seed','parameters'} & r.keys() for r in public)
    for asset in selected:
        cases = [r for r in public if r['asset_id']==asset['asset_id']]
        assert len(cases) == 3 and {r['view'] for r in cases} == {asset['view']}
        assert all(r['states']==['rest','mid','limit'] for r in cases)
    assert all(r['human_judgment']['verdict'] is None for r in rows(root/'human_review.jsonl'))


def test_single_corruption_isolation_preserves_motion_mechanism():
    gold = rows(DEFAULT_OUT/'private/gold.jsonl')
    public = {r['case_id']:r for r in rows(DEFAULT_OUT/'review_manifest.jsonl')}
    for candidate in gold:
        aid = public[candidate['case_id']]['asset_id']
        clean = next(r for r in gold if r['kind']=='clean' and public[r['case_id']]['asset_id']==aid)
        ref = json.loads((DEFAULT_OUT/f'private/{clean["case_id"]}.json').read_text())
        spec = json.loads((DEFAULT_OUT/f'private/{candidate["case_id"]}.json').read_text())
        if candidate['kind']=='part_offset':
            assert spec['joints'] == ref['joints']
            differences = [(p,q) for p,q in zip(spec['parts'],ref['parts']) if p!=q]
            assert len(differences)==1
            p,q = differences[0]
            assert {k:v for k,v in p.items() if k!='origin'} == {k:v for k,v in q.items() if k!='origin'}
        elif candidate['kind']=='joint_axis':
            assert spec['parts']==ref['parts']
            differences = [(p,q) for p,q in zip(spec['joints'],ref['joints']) if p!=q]
            assert len(differences)==1
            p,q = differences[0]
            assert {k:v for k,v in p.items() if k!='axis'} == {k:v for k,v in q.items() if k!='axis'}


def test_large_motion_difference_cannot_override_invisible_rest_offset():
    metrics = {s:{'core_changed_fraction':.8, 'largest_core_region_px':10000, 'changed_fraction':.8} for s in ('rest','mid','limit')}
    metrics['rest']['core_changed_fraction']=0
    checks={'code_correct':False,'checks':[{'passed':False},{'passed':True}]}
    assert 'rest_misalignment_visibility_insufficient' in gate('part_offset',metrics,checks)
