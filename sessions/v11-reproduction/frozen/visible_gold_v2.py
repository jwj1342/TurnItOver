"""Current-program parameter diagnosis gated by per-finding visual counterfactuals.

Private reference values are used only for audit/counterfactual construction.
They never appear in generator feedback. Unsupported rewrites stop the trial
for review rather than reusing stale injected labels or inventing a verdict.
"""
import ast,copy,io,math,re
from dataclasses import dataclass
import numpy as np
from PIL import Image
from batch_common import ObjectProgram,dump

PATTERNS={
 'position':r'g\.position\.set\(([^;]+)\);',
 'rotation':r'g\.rotation\.set\(([^;]+)\);',
 'pivot':r'pivot\.position\.set\(([^;]+)\);',
 'axis':r'const axis = new THREE\.Vector3\((.*?)\)(?:\.normalize\(\))?;',
 'limits':r'const limits:\s*\[number, number\]\s*=\s*\[([^]]+)\];',
 'sign':r'const sign = ([^;]+);',
 'anchor':r'const anchor = new THREE\.Vector3\(([^;]+)\);',
 'frame':r'const frame = new THREE\.Quaternion\(\)\.setFromEuler\(new THREE\.Euler\(([^;]+)\)\);',
}
# Restrict numeric expressions to arithmetic constants; never execute code.
def number(s):
    node=ast.parse(s.strip().replace('Math.PI','pi'),mode='eval').body
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return float(n.value)
        if isinstance(n,ast.Name) and n.id=='pi':return math.pi
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):return (-1 if isinstance(n.op,ast.USub) else 1)*visit(n.operand)
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=visit(n.left),visit(n.right)
            return a+b if isinstance(n.op,ast.Add) else a-b if isinstance(n.op,ast.Sub) else a*b if isinstance(n.op,ast.Mult) else a/b
        raise ValueError('Gold requires auditable numeric parameter expressions')
    value=visit(node)
    if not math.isfinite(value):raise ValueError('Nonfinite Gold parameter')
    return value

def parse(source):
    parts={};stripped=source
    for block in re.finditer(r'  \{ // part ([^\n]+)\n(.*?)\n  \}',source,re.S):
        name=block[1];body=block[2];values={};spans={}
        for key,pattern in PATTERNS.items():
            matches=list(re.finditer(pattern,body))
            if len(matches)>1:raise ValueError('Ambiguous Gold '+key)
            if not matches:continue
            m=matches[0];raw=m[1]
            values[key]=tuple(number(x) for x in raw.split(','))
            if key=='axis' and '.normalize()' not in m[0] and abs(np.linalg.norm(values[key])-1)>1e-9:
                raise ValueError('Unnormalized nonunit axis requires Gold review')
            spans[key]=(block.start(2)+m.start(),block.start(2)+m.end())
            stripped=stripped.replace(m[0],'',1)
        joint=re.search(r"joints\['([^']+)'\] = \{ type: '([^']+)'",body)
        if 'position' not in values:raise ValueError('Gold requires explicit part origin')
        values.setdefault('rotation',(0.,0.,0.))
        expected={'position':3,'rotation':3,'pivot':3,'axis':3,'limits':2,'sign':1,'anchor':3,'frame':3}
        if any(len(v)!=expected[k] for k,v in values.items()):raise ValueError('Unexpected Gold parameter dimensions')
        parts[name]={'values':values,'spans':spans,'joint':joint[1] if joint else None,'joint_type':joint[2] if joint else None}
    if not parts:raise ValueError('Unrecognized generated-program scaffold')
    skeleton=re.sub(r'\s+','',re.sub(r'//[^\n]*','',stripped))
    return parts,skeleton

def format_parameter(key,v):
    s=', '.join(repr(float(x)) for x in v)
    return {'position':f'g.position.set({s});','rotation':f'g.rotation.set({s});',
       'pivot':f'pivot.position.set({s});','axis':f'const axis = new THREE.Vector3({s}).normalize();',
       'limits':f'const limits: [number, number] = [{s}];','sign':f'const sign = {s};',
       'anchor':f'const anchor = new THREE.Vector3({s});',
       'frame':f'const frame = new THREE.Quaternion().setFromEuler(new THREE.Euler({s}));'}[key]

def patch(source,parts,name,updates):
    p=parts[name];edits=[]
    for key,v in updates.items():
        text=format_parameter(key,v)
        if key in p['spans']:start,end=p['spans'][key]
        elif key=='rotation':start=end=p['spans']['position'][0];text+='\n    '
        else:raise ValueError('Missing parameter in Gold counterfactual')
        edits.append((start,end,text))
    for start,end,text in sorted(edits,reverse=True):source=source[:start]+text+source[end:]
    return source

def matrix(v):
    x,y,z=v;cx,sx=math.cos(x),math.sin(x);cy,sy=math.cos(y),math.sin(y);cz,sz=math.cos(z),math.sin(z)
    return np.array([[cy*cz,-cy*sz,sy],[cx*sz+sx*sy*cz,cx*cz-sx*sy*sz,-sx*cy],[sx*sz-cx*sy*cz,sx*cz+cx*sy*sz,cx*cy]])
def euler(m):
    y=math.asin(float(np.clip(m[0,2],-1,1)))
    if abs(m[0,2])<.9999999:x=math.atan2(-m[1,2],m[2,2]);z=math.atan2(-m[0,1],m[0,0])
    else:x=math.atan2(m[2,1],m[1,1]);z=0.
    return x,y,z

def deviations(source,reference):
    actual,skel=parse(source);expected,rskel=parse(reference)
    if skel!=rskel or actual.keys()!=expected.keys():raise ValueError('Gold oracle needs review of changes outside the audited parameter scaffold')
    findings=[]
    def different(a,b):return not np.allclose(a,b,rtol=0,atol=1e-9)
    for name,part in actual.items():
        a=part['values'];b=expected[name]['values'];joint=part['joint']
        def add(kind,updates):
            findings.append({'defect_id':kind,'parts':[name],'joint_id':joint if kind=='kinematics.joint_axis' else None,
               'counterfactual':patch(source,actual,name,updates),'private_parameter_updates':updates})
        if not joint:
            if different(matrix(a['rotation']),matrix(b['rotation'])):add('structure.part_rotation',{'rotation':b['rotation']})
            if different(a['position'],b['position']):add('structure.part_offset',{'position':b['position']})
            continue
        # Diagnose physical rest pose and parent-frame motion axis, invariant
        # to equivalent decompositions among joint frame, axis, and local pose.
        fa,fb=matrix(a['frame']),matrix(b['frame'])
        rotation=fa@matrix(a['rotation']);refrotation=fb@matrix(b['rotation'])
        if different(rotation,refrotation):add('structure.part_rotation',{'rotation':euler(fa.T@refrotation)})
        origin=np.array(a['pivot'])+fa@a['position'];reforigin=np.array(b['pivot'])+fb@b['position']
        if different(origin,reforigin):add('structure.part_offset',{'position':fa.T@(reforigin-np.array(a['pivot']))})
        if different(a['pivot'],b['pivot']) or (part['joint_type']=='prismatic' and different(a['anchor'],b['anchor'])):
            add('kinematics.joint_anchor',{'pivot':b['pivot'],'anchor':b['anchor'],'position':fa.T@(origin-np.array(b['pivot']))})
        av=fa@a['axis'];bv=fb@b['axis'];an=np.linalg.norm(av);bn=np.linalg.norm(bv)
        if an==0 or bn==0:raise ValueError('Gold cannot assign a valid direction to a zero axis')
        av/=an;bv/=bn;cross=np.linalg.norm(np.cross(av,bv));dot=float(av@bv)
        if cross>1e-6:
            add('kinematics.joint_axis',{'axis':fa.T@bv})
        elif dot*a['sign'][0]*b['sign'][0]<0:
            add('kinematics.direction_flip',{'axis':fa.T@bv,'sign':(math.copysign(abs(a['sign'][0]),b['sign'][0]),)})
        if different(a['limits'],b['limits']) or abs(abs(a['sign'][0])-abs(b['sign'][0]))>1e-9:
            add('kinematics.joint_range',{'limits':b['limits'],'sign':(math.copysign(abs(b['sign'][0]),a['sign'][0]),)})
    return findings

DESCRIPTIONS={
 'structure.part_offset':('The part is displaced relative to its reference position.','Correct the part position to match the reference.'),
 'structure.part_rotation':('The part orientation differs from the reference.','Correct the part orientation.'),
 'kinematics.joint_axis':('The part moves along a different joint axis from the reference.','Correct the articulation axis and its coordinate frame.'),
 'kinematics.direction_flip':('The part moves in the opposite direction from the reference.','Restore the reference articulation direction.'),
 'kinematics.joint_anchor':('The articulation pivot is displaced from the reference.','Correct the joint pivot while preserving the intended part position.'),
 'kinematics.joint_range':('The articulation range differs from the reference.','Restore the reference articulation range.'),
}
def exposure(x,y):
    delta=np.abs(x.astype(float)-y.astype(float))/255
    mae=float(delta.mean());fraction=float((delta.max(axis=2)>.02).mean())
    return {'mae':mae,'changed_fraction':fraction,'visible':mae>=.001 and fraction>=.001}

def diagnose(source,reference,session,case,framing,evidence):
    hypotheses=deviations(source,reference);pairs=evidence['pairs']
    folder=evidence['public_record']['request'].parent
    current=[np.asarray(Image.open(p.candidate).convert('RGB')) for p in pairs]
    ref=[np.asarray(Image.open(p.reference).convert('RGB')) for p in pairs]
    visible_current=[exposure(x,y)['visible'] for x,y in zip(current,ref)]
    findings=[];audits=[]
    for index,h in enumerate(hypotheses):
        session.load(ObjectProgram(h['counterfactual']),framing=framing);scores=[]
        try:
            for i,cell in enumerate(case['observation_cells']):
                session.reset_joints();session.set_joints({case['target_joint_id']:case['state_values'][cell['state']]})
                png=session.request_view(cell['view_id']).png
                image=np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))
                scores.append(exposure(current[i],image))
        finally:session.dispose()
        steps=[i for i,row in enumerate(scores) if row['visible'] and visible_current[i]]
        audit={k:v for k,v in h.items() if k!='counterfactual'}
        audit['private_parameter_updates']={k:list(v) for k,v in audit['private_parameter_updates'].items()}
        audit.update(scores=scores,evidence_steps=steps);audits.append(audit)
        if steps:
            description,fix=DESCRIPTIONS[h['defect_id']]
            findings.append({k:h[k] for k in ('defect_id','parts','joint_id')}|{'severity':.9,'confidence':1.,'evidence_steps':steps,'description':description,'suggested_fix':fix})
    dump(folder/'gold-counterfactual-audit.private.json',{'hypotheses':audits,'visibility_proxy':'current versus per-finding counterfactual; additionally current versus reference visible','thresholds':{'mae':.001,'changed_fraction':.001,'pixel_delta':.02}})
    # A known but currently invisible discrepancy is uncertain, never certified
    # pass. Public wording discloses no hidden finding or private test result.
    status='fail' if findings else 'uncertain' if hypotheses or any(visible_current) else 'pass'
    return {'status':status,'feedback':{'status':status,'confidence':1. if status!='uncertain' else 0.,'findings':findings,
       'summary':'Visible observations support the listed diagnoses.' if findings else 'No supported diagnosis can be established from these observations.' if status=='uncertain' else 'The program matches the reference within the observed task.',
       'limitations':['Evidence support uses a frozen counterfactual pixel criterion.']},
       'observations':[{'step':i,'state':p.state} for i,p in enumerate(pairs)],'usage':{}}
