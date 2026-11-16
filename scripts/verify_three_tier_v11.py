"""Integrity and loader/scorer verification; no model training or inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import zipfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turnitover.oracle.sft_data import load_training_examples
from turnitover.oracle.sft_metrics import score_prediction, summarize
from turnitover.oracle.supervision_validation import audit_split_identities

def sha(data):return hashlib.sha256(data).hexdigest()

def verify(root,parent_archive):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        p=(root/name).resolve()
        assert p.is_relative_to(root.resolve())
        assert sha(p.read_bytes())==digest,name
    audit_split_identities(root,'episodes',manifest)
    total=0
    with zipfile.ZipFile(parent_archive) as z:
        for split in ('train','validation','test'):
            for kind in ('episodes','stopping'):
                old=[json.loads(x) for x in z.read(f'three-tier-900-sft-v1/episode_export/{split}.{kind}.jsonl').splitlines()]
                new=[json.loads(x) for x in (root/f'{split}.{kind}.jsonl').read_text().splitlines()]
                assert len(old)==len(new)
                for a,b in zip(old,new):
                    assert a['id']==b['id']
                    for key in ('images','image_sha256','data_identity','evidence','difficulty'):
                        assert a[key]==b[key]
                    assert a['messages'][0]==b['messages'][0]
                    if not b['is_clean']: assert a['messages'][1]==b['messages'][1]
                    total+=1
            for path in [x for x in z.namelist() if x.startswith('three-tier-900-sft-v1/episode_export/images/')]:
                relative=path.split('/episode_export/',1)[1]
                assert z.read(path)==(root/relative).read_bytes()
    report={'kind':'data_integrity_and_synthetic_scorer_checks_not_model_results',
            'verified_files':len(manifest['files']),'unchanged_public_records':total,
            'unchanged_image_files':sum(k.startswith('images/') for k in manifest['files']),
            'object_split_leakage':False,'splits':{}}
    # Test is mechanically converted and hashed above, never used in this loader/scoring exercise.
    for split in ('train','validation'):
        modes={mode:[r for r,_ in load_training_examples(root,split,'diagnosis',mode)] for mode in ('ordinary','evidence')}
        assert len(modes['evidence'])==manifest['label_revision_counts'][split]['eligible_cases']
        for a,b in zip(modes['ordinary'],modes['evidence']):
            assert a['messages'][0]==b['messages'][0] and a['images']==b['images']
            if a['is_clean']:assert a['messages'][1]==b['messages'][1]
        stop={mode:[r for r,_ in load_training_examples(root,split,'stopping',mode)] for mode in ('ordinary','evidence')}
        assert stop['ordinary']==stop['evidence']
        evidence=modes['evidence']
        oracle=summarize([score_prediction(r,r['messages'][1]['content']) for r in evidence],defect_ids=manifest['experiment_taxonomy'])
        assert oracle['status_accuracy']==oracle['exact_verdict_accuracy']==1
        assert oracle['clean_pass_rate']==1 and oracle['clean_false_positive_rate']==0 and oracle['false_pass_rate']==0
        fake_pass=json.dumps({'verdict':{'status':'pass','confidence':1,'findings':[],'summary':'Synthetic test only.','limitations':[]}})
        always_pass=summarize([score_prediction(r,fake_pass) for r in evidence],defect_ids=manifest['experiment_taxonomy'])
        assert always_pass['false_pass_rate']==1 and always_pass['defective_false_pass_rate']==1
        fake_uncertain=json.dumps({'verdict':{'status':'uncertain','confidence':0,'findings':[],'summary':'Synthetic test only.','limitations':['No conclusion.']}})
        abstain=summarize([score_prediction(r,fake_uncertain) for r in evidence],defect_ids=manifest['experiment_taxonomy'])
        assert abstain['clean_pass_rate']==0 and abstain['abstention_rate']==1
        report['splits'][split]={'loaded_cases':len(evidence),'oracle_status_accuracy':oracle['status_accuracy'],
            'oracle_clean_cases':oracle['clean_cases'],'always_pass_defective_false_pass_rate':always_pass['defective_false_pass_rate'],
            'always_uncertain_clean_pass_rate':abstain['clean_pass_rate'],'expected_uncertain_cases':oracle['expected_uncertain_cases'],
            'ordinary_evidence_identical_public_inputs':True,'stopping_conditions_identical':True}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--parent-archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=verify(a.dataset,a.parent_archive)
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
