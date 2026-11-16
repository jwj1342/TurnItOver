"""Aggregate completed, replay-verified cases, with source-object bootstrap."""
from pathlib import Path
import json,csv,hashlib,collections
import numpy as np
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def estimate(values,ids,draws):
    # Row values are already seed-averaged per case. Bootstrap whole objects.
    point=float(np.mean(values))
    means=[]
    for draw in draws:
        sample=[values[i] for obj in draw for i in ids.get(int(obj),[])]
        if sample:means.append(float(np.mean(sample)))
    lo,hi=np.quantile(means,[.025,.975],method='linear')
    return {'value':point,'ci95':[float(lo),float(hi)],'valid_bootstrap_samples':len(means),'cases':len(values)}

def main():
    protocol=load(ROOT/'protocol.json');cases=load(ROOT/'selection.json')
    objects=sorted({c['data_identity']['split_group'] for c in cases});objindex={s:i for i,s in enumerate(objects)}
    draws=np.random.default_rng(protocol['bootstrap_seed']).integers(0,len(objects),(1000,len(objects)))
    outcomes=[];audits=[];grid_count=0;full_visible=0;case_info=[]
    for case in cases:
        d=ROOT/'results'/case['case_id'];audit=load(d/'replay.json')
        assert audit['status']=='complete' and audit['all_replayed_images_equal_grid']
        grid=load(d/'grid.json');results=load(d/'outcomes.json');traces=load(d/'trajectories.json')
        assert len(results)==408
        assert set(r['trace_id'] for r in results)==set(traces)
        assert audit['unique_trajectories_replayed']==len(traces)
        for r in results:
            trace=traces[r['trace_id']]
            first=next((i+1 for i,a in enumerate(trace) if grid['signals'][a['witness_index']]['exposed']),r['budget']+1)
            assert first==r['first_visible'] and int(first<=r['budget'])==r['exposed']
        outcomes+=results;audits.append(audit);grid_count+=len(grid['grid'])
        visible=any(s['exposed'] for s in grid['signals']);full_visible+=visible
        case_info.append({'case_id':case['case_id'],'object':case['data_identity']['split_group'],'defect_id':case['annotation']['labels'][0]['defect_id'],'grid_states':len(grid['grid']),'full_grid_exposed':visible})
    rows=[];per_case=[]
    for k in [1,2,4,6]:
        for policy in ['fixed','random','oracle']:
            subset=[r for r in outcomes if r['budget']==k and r['policy']==policy]
            means=[]
            for case in cases:
                entries=[r for r in subset if r['case_id']==case['case_id']]
                assert len(entries)==(100 if policy=='random' else 1)
                means.append({'case_id':case['case_id'],'object_index':objindex[case['data_identity']['split_group']],
                    'defect_id':case['annotation']['labels'][0]['defect_id'],
                    'exposed':float(np.mean([r['exposed'] for r in entries])),
                    'first_visible':float(np.mean([r['first_visible'] for r in entries])),
                    'actions_used':float(np.mean([r['actions_used'] for r in entries]))})
            per_case.extend(dict(m,budget=k,policy=policy) for m in means)
            row={'budget':k,'policy':policy}
            for metric,field,category in [('structure_recall','exposed','structure.'),('kinematics_recall','exposed','kinematics.'),('all_recall','exposed',''),('case_coverage','exposed',''),('first_visible','first_visible','')]:
                ms=[m for m in means if m['defect_id'].startswith(category)]
                ids=collections.defaultdict(list)
                for i,m in enumerate(ms):ids[m['object_index']].append(i)
                row[metric]=estimate([m[field] for m in ms],ids,draws)
            rows.append(row)
    effects=[]
    for k in [1,2,4,6]:
        for baseline in ['fixed','random']:
            paired=[];ids=collections.defaultdict(list)
            for i,case in enumerate(cases):
                oracle=next(r for r in per_case if r['budget']==k and r['policy']=='oracle' and r['case_id']==case['case_id'])
                other=next(r for r in per_case if r['budget']==k and r['policy']==baseline and r['case_id']==case['case_id'])
                paired.append(oracle['exposed']-other['exposed']);ids[oracle['object_index']].append(i)
            effects.append({'budget':k,'contrast':'oracle_minus_'+baseline,'recall_difference':estimate(paired,ids,draws)})
    result={'status':'complete','cases':len(cases),'objects':len(objects),'protocol':protocol,'rows':rows,'paired_effects':effects,
      'full_grid_exposed_cases':full_visible,'full_grid_states':grid_count,'rendered_grid_images':grid_count*2,
      'reported_policy_runs':len(outcomes),'unique_trajectories_replayed':sum(a['unique_trajectories_replayed'] for a in audits),
      'replayed_actions':sum(a['replayed_actions'] for a in audits),'case_info':case_info,
      'ci_method':'1000 source-object bootstrap draws; Random averaged within case before bootstrap; percentile with linear interpolation; category-empty draws omitted only for that category',
      'limitations':['Single-defect eligible subset: 19 Easy and 15 Medium, no Hard; three represented defect types.',
      'Pixel exposure proxy, not human recognizability or model diagnosis.',
      'All recall equals case coverage on this single-defect subset.',
      'Five structural cases yield limited category precision; degenerate empirical bootstrap intervals do not imply population certainty.',
      'Privileged greedy on a finite grid is not a global optimum or learned active policy.']}
    dump(ROOT/'summary.json',result);dump(ROOT/'per-case-metrics.json',per_case)
    with (ROOT/'table3.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['budget','policy']+[f'{m}_{s}' for m in ['structure_recall','kinematics_recall','all_recall','case_coverage','first_visible'] for s in ['value','ci_low','ci_high']]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            out={k:r[k] for k in ['budget','policy']}
            for m in ['structure_recall','kinematics_recall','all_recall','case_coverage','first_visible']:
                out.update({f'{m}_value':r[m]['value'],f'{m}_ci_low':r[m]['ci95'][0],f'{m}_ci_high':r[m]['ci95'][1]})
            writer.writerow(out)
    metrics=['structure_recall','kinematics_recall','all_recall','case_coverage','first_visible']
    def fmt(m,x):
        scale=1 if m=='first_visible' else 100;dec=2 if m=='first_visible' else 1
        return f"{x['value']*scale:.{dec}f} [{x['ci95'][0]*scale:.{dec}f}, {x['ci95'][1]*scale:.{dec}f}]"
    labels={'fixed':'Fixed','random':'Random','oracle':'Greedy oracle'}
    caption=(r'\textbf{Observation selection under limited budgets.} '
        r'We evaluate all 34 eligible single-defect validation cases from \texttt{three-tier-900-sft-v1.1}, representing 34 objects '
        r'(19 Easy, 15 Medium; 5 part offsets, 12 joint-axis errors, and 17 direction flips). '
        r'Recall and coverage are percentages under a paired-image pixel-exposure proxy; each interval is a 95\% source-object bootstrap interval (1,000 draws). '
        r'Random averages 100 orders per case. All policies pay one action for the initial front view; restoring a joint and changing a view also each cost one. '
        r'First exposure assigns $K+1$ to failures. All recall and case coverage coincide because each case has one defect. '
        r'The oracle uses privileged visibility annotations; these results do not measure learned action selection or diagnostic accuracy.')
    tex=['\\begin{table*}[t]','  \\centering','  \\scriptsize','  \\setlength{\\tabcolsep}{3.2pt}','  \\caption{'+caption+'}','  \\label{tab:oracle_selection}',r'  \begin{tabular}{@{}clccccc@{}}',r'    \toprule',r'    Budget $K$ & Policy & Structure recall $\uparrow$ & Kinematics recall $\uparrow$ & All recall $\uparrow$ & Case coverage $\uparrow$ & First exposure $\downarrow$ \\',r'    \midrule']
    for k in [1,2,4,6]:
        for idx,r in enumerate(x for x in rows if x['budget']==k):
            left=f'\\multirow{{3}}{{*}}{{{k}}}' if idx==0 else ''
            tex.append('    '+left+' & '+labels[r['policy']]+' & '+' & '.join(fmt(m,r[m]) for m in metrics)+r' \\')
        if k!=6:tex.append(r'    \midrule')
    tex.extend([r'    \bottomrule',r'  \end{tabular}',r'\end{table*}',''])
    (ROOT/'table3.tex').write_text('\n'.join(tex),encoding='utf-8')
    report=['# Table 3 实测结果','',f"已完成 {len(cases)} 个案例／{len(objects)} 个来源物体；{result['unique_trajectories_replayed']} 条不同轨迹、{result['replayed_actions']} 次动作通过实际重放图像一致性检查。",'',
       '| K | Policy | Structure % | Kinematics % | All % | Coverage % | First exposure |','|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:report.append('| '+str(r['budget'])+' | '+labels[r['policy']]+' | '+' | '.join(fmt(m,r[m]) for m in metrics)+' |')
    report+=['','区间：按来源物体 bootstrap 1,000 次，Random 先在每例内平均 100 个顺序；不是把随机顺序当独立物体。',
      '',f'完整网格暴露 {full_visible}/{len(cases)} 例；不可见案例仍保留在分母。','',
      '范围：v1.1 全部准入单缺陷 validation，19 Easy、15 Medium；5 part_offset、12 joint_axis、17 direction_flip。排除 10 个复合案例，未使用 Test 或训练模型。',
      '', '结果衡量像素暴露代理；不能表述为模型识别或原生 2B 主动观察表现。单缺陷子集使 All recall 与 Coverage 相同，Structure 只有 5 个物体。',
      '', '所有新产物位于本目录；服务器仅只读下载。',
      '', '复现：`python -X utf8 run_table3.py`，完成后 `python -X utf8 summarize.py`。使用已下载的 v1.1 runtime 与预编译 Three.js harness、本机 Chromium 1187。']
    (ROOT/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['cases','objects','full_grid_exposed_cases','unique_trajectories_replayed','replayed_actions']},indent=2))
    print('\n'.join(report[4:18]))
if __name__=='__main__':main()
