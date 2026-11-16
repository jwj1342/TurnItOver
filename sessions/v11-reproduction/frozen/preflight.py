"""Render and independently audit all 44 cases, without any model API calls."""
import argparse,sys,traceback
from pathlib import Path
from dataclasses import asdict
from batch_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('batch',type=Path);p.add_argument('--attempt',default='v1');a=p.parse_args()
    sys.path.insert(0,str(a.batch/'code'))
    from types import SimpleNamespace
    def pair(state,reference,candidate):return SimpleNamespace(state=state,reference=reference,candidate=candidate)
    dataset,records,cases,cfg=setup(a.root,a.batch)
    out=a.batch/('preflight-'+a.attempt);out.mkdir(exist_ok=True)
    reports=[]
    with ObservationSession(cfg.render,load_views(cfg.views_path)) as session:
        for case in cases:
            uid=case['case_id'];folder=out/uid;folder.mkdir(exist_ok=True)
            report={'case_id':uid,'difficulty':case['difficulty'],'image_count':len(records[uid]['images'])}
            try:
                ref,initial=sources(a.batch,case);reference=ObjectProgram(ref)
                framing,attempts=recover_framing(session,reference,case,records[uid],dataset,initial)
                report.update(framing=asdict(framing),camera_attempts=attempts)
                observe=observer(session,records[uid],case,reference,framing,pair)
                evidence=observe(initial,folder)
                assert 'gate' not in evidence
                checks=[]
                for name in records[uid]['images']:
                    frozen=np.asarray(Image.open(dataset/name).convert('RGB'))
                    fresh=np.asarray(Image.open(folder/Path(name).name).convert('RGB'))
                    checks.append({'image':name,'pixels_identical':np.array_equal(frozen,fresh),'mae':float(np.abs(frozen.astype(float)-fresh).mean())})
                report['image_checks']=checks
                assert all(row['pixels_identical'] for row in checks),'Frozen image replay mismatch'
                audit=RepairEnvironment(session,reference,framing,cfg,{})
                from turnitover.checkers.runtime_budget import RuntimeBudgetChecker
                limits={'max_triangles':int(audit.gold.stats['triangles_scene']),'max_draw_calls':int(audit.gold.stats['draw_calls'])}
                audit.checkers=(audit.checkers[0],RuntimeBudgetChecker(**limits))
                report['runtime_policy']={'kind':'reference_complexity','status':'confirmed',**limits}
                control=audit.audit(ref);candidate=audit.audit(initial)
                dump(folder/'reference.audit.private.json',control);dump(folder/'candidate.audit.private.json',candidate)
                report.update(reference_passed=control['passed'],candidate_usable=candidate['usable'],candidate_initial_passed=candidate['passed'])
                report['status']='eligible' if control['passed'] and candidate['usable'] and not candidate['passed'] else 'eligibility_check_failed'
            except Exception as exc:
                report.update(status='preflight_error',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            reports.append(report);dump(folder/'preflight.json',report);dump(out/'summary.json',reports)
            print('PREFLIGHT',len(reports),uid,report['status'],report.get('error',''),flush=True)
    print('ALL_44_PREFLIGHT_COMPLETE',flush=True)
if __name__=='__main__':main()
