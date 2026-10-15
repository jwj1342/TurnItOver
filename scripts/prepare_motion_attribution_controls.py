"""Bounded six-case grounding control and exact historical closed-loop replay."""
import argparse
import copy
import json
from pathlib import Path
from scripts.build_motion_attribution import dump
from scripts.build_visual_decisions import sha
from scripts.evaluate_modular_verifier import _read_jsonl, _templates
from turnitover.verifier.modular import build_single_reference_record


def main(root):
    rows=[]
    original=_read_jsonl(root/'inputs.jsonl')
    for row in original:
        if row['condition']!='B' or row['asset_id'] not in ('PM_100202','PM_102252'): continue
        d=copy.deepcopy(row)
        structure=json.loads(row['messages'][0]['content'].split('\n')[-1])['structure']
        # Sheet is composed exclusively of the same twelve B images, no Gold overlay.
        sheet=Path(row['images'][1]).parent/'review.png'
        d.update(condition='D',images=[str(sheet)],image_sha256=[sha(sheet)])
        prompt='''The image is a contact sheet. TOP ROW is REFERENCE, BOTTOM ROW is CANDIDATE.
Each column is one matched camera and joint angle; columns 0,1,2 are front rest/mid/end,
columns 3,4,5 are oblique rest/mid/end. Compare TOP versus BOTTOM within each column,
then compare how the SAME part changes across columns. Ordinary rotation is NOT an error.
Choose: clean = candidate matches reference at all shown states; axis = candidate rotates
in a DIFFERENT plane/direction from reference; offset = displaced part position.
Choose uncertain if you cannot tell. Do not assume that motion implies an axis error.
Return JSON: abnormal (true/false/null), classification (axis/offset/clean/uncertain),
joint_id, part_id, evidence_steps (column numbers), explanation (visible comparison).
Choose IDs from the public structure; null IDs for clean/uncertain and null joint for offset.
Public structure: '''+json.dumps(structure)
        d['messages']=[{'role':'user','content':prompt},{'role':'assistant','content':''}];rows.append(d)
    (root/'controls.inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    v4=Path('../TurnItOver-dataset/data/benchmarks/partnet-mobility-formal-v1/oracle-first-failure-v4').resolve()
    templates,_=_templates(v4)
    historic=Path('output/partnet-modular-verifier-v4-only-v1').resolve()
    replay=[];audit=[]
    for ep in _read_jsonl(historic/'episodes.jsonl'):
        if ep['policy']!='fixed':continue
        uid=ep['source_sample_id'];folder=historic/'episodes'/f'{uid}-fixed'
        # Last actual diagnosis call, with precisely its then-visible history and remaining budget.
        call=next(c for c in reversed(ep['calls']) if c.get('diagnosis'))
        n=call['step'];h=ep['history'][:n]
        row=build_single_reference_record(templates[uid],h,6-n)
        row.update(id=uid,asset_id=ep['asset_id'],condition='A_original',
                   images=[str(v4/templates[uid]['images'][0]),*[str(folder/'observations'/f'{i:03d}.png') for i in range(n)]])
        row['messages'][1]['content']='';row['image_sha256']=[sha(Path(p)) for p in row['images']]
        replay.append(row);audit.append({'id':uid,'original_raw':call['diagnosis']['raw'],
              'original_input_tokens':call['diagnosis']['input_tokens'],'history_count':n,
              'source_episode':str(folder/'episode.json'),'gold_stratum':ep['gold_stratum'],
              'expected_visible_findings':ep['expected_visible_findings']})
    (root/'historical.inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in replay))
    dump(root/'historical.audit.json',audit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
