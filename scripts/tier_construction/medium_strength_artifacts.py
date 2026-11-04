"""Native artifacts, review pages and checkpoints for the revised Medium set."""
from __future__ import annotations

from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import random

from scripts.tier_construction.construction_pilot import write_frames
from scripts.tier_construction.prepare_level0_axis_review import board, pair_preview
from scripts.tier_construction.prepare_medium_pilot import _json, _jsonl, _review
from scripts.tier_construction.prepare_obvious_defect_review import STATES, checker_result, sha256_path
from turnitover.core.serde import to_dict

SEED = 2026102321
TARGET = 75


def object_id(configuration, asset_id):
    return hashlib.sha256(f'medium-strength-v3:{configuration}:{asset_id}'.encode()).hexdigest()[:16]


def artifacts(out, item, pair, reference):
    ref, chosen, joint_id, values = pair
    configuration, view = item['config'], chosen['view']
    source = {'split': 'train', **item['metadata'], 'source_urdf': item['source_urdf'],
              'complexity': item['complexity']}
    rows = {key: [] for key in ('public', 'gold', 'checks', 'reviews')}
    for kind, program, rendered, label, seed in (
            ('clean', reference, ref, None, None),
            (chosen['label'].defect_id.split('.')[1], chosen['program'], chosen['render'], chosen['label'], chosen['seed'])):
        raw = f'medium-strength-v3:{configuration}:{reference.spec.asset_id}:{kind}:{seed}'
        cid = 'ms-' + hashlib.sha256(raw.encode()).hexdigest()[:14]
        result = checker_result(program, rendered, reference, ref)
        if result['code_correct'] != (label is None) or not result['checks'][1]['passed']:
            raise ValueError(f'Export checker disagreement: {cid}')
        board(out / f'boards/{cid}.png', cid, ref, rendered, view)
        pair_preview(out / f'inputs/{cid}.png', ref, rendered, view)
        write_frames(out, cid, ref, rendered, view)
        for role, frames in (('reference', ref), ('candidate', rendered)):
            for state in STATES:
                path = out / f'native/{cid}/{role}_{state}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(frames['frames'][state][view])
        program_path = f'private/programs/{cid}.json'
        _json(out / program_path, to_dict(program.spec))
        rows['public'].append({'case_id': cid, 'category': reference.spec.category, 'view': view,
            'states': list(STATES), 'input_states': ['rest', 'limit'], 'image_size': [256, 256],
            'board': f'boards/{cid}.png', 'input_preview': f'inputs/{cid}.png',
            'images': [f'frames/{cid}/{i}.png' for i in range(4)], 'status': 'pending_human_review'})
        rows['gold'].append({'case_id': cid, 'asset_id': reference.spec.asset_id, 'config': configuration,
            'difficulty': 'medium_provisional', 'screening_policy': 'medium-strength-v3', 'kind': kind,
            'expected_status': 'pass' if label is None else 'fail', 'defect_id': None if label is None else label.defect_id,
            'joint_id': joint_id, 'parts': [] if label is None else list(label.parts),
            'parameters': {} if label is None else label.params, 'seed': seed, 'state_values': values,
            'candidate_sha': program.sha, 'reference_sha': reference.sha, 'program': program_path,
            'source': source, 'training_eligibility': 'pending_human_review'})
        rows['checks'].append({'case_id': cid, **result})
        rows['reviews'].append(_review(cid))
    return rows


def page(out, public, name):
    cards = []
    for row in public:
        cid = html.escape(row['case_id'])
        cards.append(f'<section data-id="{cid}"><h2>{cid}</h2>'
            '<div class="motion"><figure><figcaption>参考</figcaption>'
            f'<img class="reference" src="native/{cid}/reference_rest.png"></figure>'
            '<figure><figcaption>候选</figcaption>'
            f'<img class="candidate" src="native/{cid}/candidate_rest.png"></figure></div>'
            '<p class="state">rest</p><button class="play">播放三态</button> '
            '<button class="step">下一状态</button>'
            f'<details><summary>三态并排对比</summary><img class="board" src="{html.escape(row["board"])}"></details>'
            '<p>缺陷可见性 <select class="difficulty"><option value="">请选择</option>'
            '<option value="medium">仔细对比可发现，符合 Medium</option>'
            '<option value="too_obvious">一眼明显，应排除</option>'
            '<option value="too_subtle">难以发现，应排除</option>'
            '<option value="clean">未发现缺陷</option><option value="invalid">遮挡、出框或其他问题</option></select></p>'
            '<textarea placeholder="观察到的部件、状态、运动或位置变化（请填写具体证据）"></textarea></section>')
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Medium 强度人工审核</title>
<style>body{font:16px system-ui;max-width:960px;margin:32px auto;padding:0 16px;color:#18212b;background:#f7f8fa}
section{background:white;padding:20px;margin:28px 0;border:1px solid #ddd;border-radius:10px}
h2{font-size:17px}figure{margin:0}figcaption{text-align:center}.motion{display:flex;gap:16px;flex-wrap:wrap}
.motion img{width:256px;height:256px}.board{max-width:100%}button,select{padding:8px}textarea{width:95%;height:70px}
.state{color:#526070}header{position:sticky;top:0;background:#f7f8fa;padding:10px 0;z-index:1}</style>
<header><h1>Medium 强度人工审核</h1><button id="export">导出本页审核 JSON</button></header>
<p>参考与候选使用同一相机、同一关节状态和原生 256×256 图像。播放按 rest → mid → limit → mid 循环，
是三个实际渲染状态的切换。请先观察运动，再展开并排图仔细比较。</p>
<p>验收：缺陷真实存在，仔细比较或观察运动能够发现；排除一眼夸张和难以辨认的样本。
自动校验和助手视觉筛选均不等于人工通过。选择和备注仅保存在本浏览器，请导出审核文件。</p>
''' + '\n'.join(cards) + '''<script>
const states=['rest','mid','limit','mid'];
document.querySelectorAll('section').forEach(s=>{
let index=0,timer=null; const id=s.dataset.id,key='medium-strength-v3:'+id;
const show=()=>{const state=states[index];['reference','candidate'].forEach(role=>s.querySelector('.'+role).src='native/'+id+'/'+role+'_'+state+'.png');s.querySelector('.state').textContent=state;};
s.querySelector('.step').onclick=()=>{index=(index+1)%4;show();};
s.querySelector('.play').onclick=e=>{if(timer){clearInterval(timer);timer=null;e.target.textContent='播放三态';}else{timer=setInterval(()=>{index=(index+1)%4;show();},700);e.target.textContent='暂停';}};
const select=s.querySelector('select'),notes=s.querySelector('textarea');
try{const v=JSON.parse(localStorage.getItem(key)||'{}');select.value=v.decision||'';notes.value=v.notes||'';}catch(e){}
const save=()=>localStorage.setItem(key,JSON.stringify({case_id:id,decision:select.value,notes:notes.value}));select.onchange=save;notes.oninput=save;
});
document.querySelector('#export').onclick=()=>{const rows=[...document.querySelectorAll('section')].map(s=>({case_id:s.dataset.id,reviewer:'human',decision:s.querySelector('select').value||null,notes:s.querySelector('textarea').value,reviewed_at:new Date().toISOString()}));
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='medium-human-review.json';a.click();URL.revokeObjectURL(a.href);};
</script></html>'''
    (out / name).write_text(document, encoding='utf-8')


def checkpoint(out, records, inputs, base, policy):
    rejected_ids = set()
    exclusions = out / 'private/visual_rejections.json'
    if exclusions.exists():
        rejected_ids = {r['asset_id'] for r in json.loads(exclusions.read_text())}
    accepted = [r for r in records if r['status'] == 'automatic_pass' and r['asset_id'] not in rejected_ids]
    public = sorted((p for r in accepted for p in r['public']), key=lambda p: p['case_id'])
    for filename, key in (('review_manifest.jsonl', 'public'), ('private/gold.jsonl', 'gold'),
                          ('private/checks.jsonl', 'checks'), ('human_review.jsonl', 'reviews')):
        _jsonl(out / filename, sorted((p for r in accepted for p in r[key]), key=lambda p: p['case_id']))
    _jsonl(out / 'train_manifest.jsonl', [])
    _jsonl(out / 'private/train_targets.jsonl', [])
    page(out, public, 'review.html')
    defects = sorted((p for r in accepted for g, p in zip(r['gold'], r['public']) if g['kind'] != 'clean'), key=lambda p: p['case_id'])
    sampled = random.Random(SEED).sample(defects, min(10, len(defects)))
    _json(out / 'spot_check_10.json', {'seed': SEED, 'case_ids': [p['case_id'] for p in sampled],
        'sampling_frame': 'one defective candidate per independent object; uniform random without replacement',
        'human_review_status': 'pending_human_review'})
    page(out, sampled, 'review-spot-10.html')
    old = json.loads((out / 'private/source_audit.json').read_text())
    old_objects = old['original_object_ids']
    by_asset = {r['asset_id']: r for r in accepted}
    _json(out / 'private/replacement_audit.json', {'original_objects': len(old_objects),
        'objects': [{'asset_id': aid, 'disposition': 'regenerated' if aid in by_asset else 'excluded_and_replaced',
                     'new_case_ids': [p['case_id'] for p in by_asset[aid]['public']] if aid in by_asset else []}
                    for aid in old_objects],
        'replacement_objects': sorted(set(by_asset) - set(old_objects))})
    report = {'status': 'automatic_candidates_ready' if len(accepted) == TARGET else 'collection_incomplete',
        'target_objects': TARGET, 'independent_objects': len(accepted), 'cases': len(public),
        'configs': dict(Counter(r['config'] for r in accepted)), 'attempted_source_configurations': len(records),
        'human_review_status': 'pending_human_review', 'human_review_complete': 0,
        'training_eligible_objects': 0, 'training_eligible_cases': 0, 'native_image_size': [256, 256],
        'model_calls': 0, 'paid_api_calls': 0, 'training_runs': 0,
        'assistant_visual_excluded_objects': sorted(rejected_ids)}
    _json(out / 'validation.json', report)
    seal(out, inputs, base, policy)
    return report


def seal(out, inputs, base, policy):
    _json(out / 'manifest.json', {'version': 2, 'policy': policy, 'target_objects': TARGET,
        'base_directory': str(base), 'inputs': inputs,
        'files': {p.relative_to(out).as_posix(): sha256_path(p) for p in sorted(out.rglob('*'))
                  if p.is_file() and p != out / 'manifest.json'}})
