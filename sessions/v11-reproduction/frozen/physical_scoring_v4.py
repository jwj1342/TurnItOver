"""Per-original-injection physical restoration, independent of other defects.

Compare in the declared parent-part coordinate frame, invariant to equivalent
joint-frame/axis/local-pose decompositions. Unknown program semantics are explicit.
"""
import numpy as np
from visible_gold_v4 import parse,matrix,canonicalize,normalize_for_reference

POSITION_ATOL=1e-9
ROTATION_MATRIX_ATOL=1e-9
AXIS_SINE_ATOL=1e-6

def score_original(source,reference,labels,*,usable=True):
    rows=[{'label_index':i,'defect_id':l['defect_id'],'parts':l['parts'],'resolved':None,
           'checker_id':'original_injection_physical_parameters_v1'} for i,l in enumerate(labels)]
    if not usable:
        for row in rows:row['unknown_reason']='Final program cannot be compiled/rendered as required.'
        return rows
    try:
        actual,skel=parse(normalize_for_reference(source,reference));expected,rskel=parse(canonicalize(reference))
        if skel!=rskel or actual.keys()!=expected.keys():raise ValueError('Changed code outside the validated physical parameter scaffold')
    except Exception as exc:
        for row in rows:row['unknown_reason']=str(exc)
        return rows
    for row,label in zip(rows,labels):
        details=[]
        try:
            for name in label['parts']:
                part=actual[name];refpart=expected[name];a,b=part['values'],refpart['values']
                joint=bool(part['joint']);fa=matrix(a['frame']) if joint else np.eye(3);fb=matrix(b['frame']) if joint else np.eye(3)
                anchor_key='anchor' if part['joint_type']=='prismatic' else 'pivot'
                aa=np.array(a[anchor_key]) if joint else np.zeros(3);ba=np.array(b[anchor_key]) if joint else np.zeros(3)
                kind=label['defect_id'];item={'part':name}
                if kind=='structure.part_offset':
                    error=float(np.max(np.abs(aa+fa@a['position']-ba-fb@b['position'])))
                    item.update(passed=error<=POSITION_ATOL,error_linf_m=error,threshold=POSITION_ATOL)
                elif kind=='structure.part_rotation':
                    error=float(np.max(np.abs(fa@matrix(a['rotation'])-fb@matrix(b['rotation']))))
                    item.update(passed=error<=ROTATION_MATRIX_ATOL,error_matrix_linf=error,threshold=ROTATION_MATRIX_ATOL)
                elif kind in ('kinematics.joint_axis','kinematics.direction_flip'):
                    if not joint or part['joint']!=refpart['joint']:raise ValueError('Required joint identity is missing')
                    av=fa@a['axis'];bv=fb@b['axis'];an=np.linalg.norm(av);bn=np.linalg.norm(bv)
                    if an<=1e-12 or bn<=1e-12:
                        item.update(passed=False,reason='Zero motion axis')
                    else:
                        av/=an;bv/=bn;sine=float(np.linalg.norm(np.cross(av,bv)));dot=float(av@bv)
                        direction=bool(dot*a['sign'][0]*b['sign'][0]>0)
                        item.update(passed=sine<=AXIS_SINE_ATOL and (kind!='kinematics.direction_flip' or direction),
                                    axis_sine_error=sine,axis_dot=dot,same_signed_direction=direction,threshold=AXIS_SINE_ATOL)
                elif kind=='kinematics.joint_anchor':
                    if part['joint_type']!='revolute':raise ValueError('No frozen prismatic-anchor restoration predicate')
                    axis=fb@b['axis'];axis/=np.linalg.norm(axis)
                    # A revolute hinge anchor is an axis line; moving its point
                    # along that same line is a physically equivalent encoding.
                    error=float(np.linalg.norm(np.cross(aa-ba,axis)))
                    item.update(passed=error<=POSITION_ATOL,reference_axis_line_distance_m=error,threshold=POSITION_ATOL)
                else:raise ValueError('No frozen physical predicate for '+kind)
                details.append(item)
            if not details:raise ValueError('Original finding has no identifiable target part')
            row.update(resolved=all(d['passed'] for d in details),measurements=details)
        except Exception as exc:row.update(unknown_reason=str(exc),measurements=details)
    return rows
