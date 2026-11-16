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

def canonicalize(source):
    """Normalize a narrowly matched equivalent inline revolute setter privately."""
    # Move a setter-only unary minus into its sole numeric sign declaration.
    def normalize_sign(block):
        body=block[0]
        if '-sign * v' not in body:return body
        declaration=re.search(PATTERNS['sign'],body)
        if not declaration or len(re.findall(r'\bsign\b',body))!=2:
            raise ValueError('Negated setter sign has unaudited dependencies')
        value=-number(declaration[1])
        body=body[:declaration.start()]+f'const sign = {value!r};'+body[declaration.end():]
        return body.replace('-sign * v','sign * v')
    source=re.sub(r'  \{ // part [^\n]+\n.*?\n  \}',normalize_sign,source,flags=re.S)
    def normalize_parent_coordinates(block):
        body=block[0]
        pattern=(r"const Rl1 = new THREE\.Matrix4\(\)\.makeRotationFromEuler\(new THREE\.Euler\(([^)]+)\)\);\s*"
                 r'const axisWorld = new THREE\.Vector3\(([^)]+)\)\.normalize\(\);\s*'
                 r'const anchorWorld = new THREE\.Vector3\(([^)]+)\);\s*'
                 r'const axis = axisWorld\.clone\(\)\.applyMatrix4\(new THREE\.Matrix4\(\)\.extractRotation\(Rl1\)\)\.normalize\(\);\s*'
                 r'const anchorLocal = anchorWorld\.clone\(\)\.applyMatrix4\(Rl1\.clone\(\)\.invert\(\)\);')
        match=re.search(pattern,body)
        if not match:return body
        clean=re.sub(r'//[^\n]*','',body)
        for name,count in [('Rl1',3),('axisWorld',2),('anchorWorld',2),('anchorLocal',2)]:
            if len(re.findall(r'\b'+name+r'\b',clean))!=count:raise ValueError('Parent transform has unaudited dependencies')
        fields=match[1].split(',')
        if len(fields)!=4 or ast.literal_eval(fields[3].strip())!='XYZ':raise ValueError('Unsupported parent rotation')
        rotation=matrix([number(x) for x in fields[:3]])
        axis=np.array([number(x) for x in match[2].split(',')]);axis/=np.linalg.norm(axis)
        anchor=rotation.T@np.array([number(x) for x in match[3].split(',')])
        body=body[:match.start()]+format_parameter('axis',rotation@axis)+'\n    '+format_parameter('anchor',anchor)+body[match.end():]
        assert body.count('pivot.position.copy(anchorLocal);')==1
        body=body.replace('pivot.position.copy(anchorLocal);','')
        pivots=list(re.finditer(PATTERNS['pivot'],body))
        if len(pivots)!=1:raise ValueError('Ambiguous transformed pivot origin')
        p=pivots[0];body=body[:p.start()]+format_parameter('pivot',anchor)+body[p.end():]
        positions=list(re.finditer(PATTERNS['position'],body))
        if len(positions)!=2:raise ValueError('Unexpected construction position writes')
        p=positions[0];body=body[:p.start()]+body[p.end():]
        return body
    source=re.sub(r'  \{ // part [^\n]+\n.*?\n  \}',normalize_parent_coordinates,source,flags=re.S)
    def normalize_pivot_expression(block):
        body=block[0]
        match=re.search(r'pivot\.position\.copy\(new THREE\.Vector3\(([^)]+)\)\.applyEuler\(g\.rotation\)\.negate\(\)\);',body)
        if not match:return body
        rotation_writes=list(re.finditer(PATTERNS['rotation'],body))
        if len(rotation_writes)>1 or rotation_writes and rotation_writes[0].end()>match.start():
            raise ValueError('Pivot rotation dependency is not a fixed construction value')
        parsed,_=parse(body);part=next(iter(parsed.values()))
        anchor=-(matrix(part['values']['rotation'])@np.array([number(x) for x in match[1].split(',')]))
        body=body[:match.start()]+format_parameter('pivot',anchor)+body[match.end():]
        if body.count('const anchor = pivot.position.clone();')!=1:raise ValueError('Unsupported pivot anchor alias')
        body=body.replace('const anchor = pivot.position.clone();',format_parameter('anchor',anchor))
        return body
    source=re.sub(r'  \{ // part [^\n]+\n.*?\n  \}',normalize_pivot_expression,source,flags=re.S)
    def normalize_rotated_axis(block):
        body=block[0]
        pattern=r'const Rparent = new THREE\.Matrix4\(\)\.makeRotationFromEuler\(new THREE\.Euler\(([^)]+)\)\)\.invert\(\);\s*const axisLocal = new THREE\.Vector3\(([^)]+)\)\.applyMatrix4\(Rparent\)\.normalize\(\);'
        match=re.search(pattern,body)
        if match:
            clean=re.sub(r'//[^\n]*','',body)
            if len(re.findall(r'\bRparent\b',clean))!=2 or len(re.findall(r'\baxisLocal\b',clean))!=2:
                raise ValueError('Rotated axis has unaudited dependencies')
            angles=[number(x) for x in match[1].split(',')]
            vector=np.array([number(x) for x in match[2].split(',')])
            if len(angles)!=3 or vector.shape!=(3,):raise ValueError('Unsupported rotated-axis dimensions')
            rotated=matrix(angles).T@vector
            body=body[:match.start()]+format_parameter('axis',rotated)+body[match.end():]
            body=body.replace('setFromAxisAngle(axisLocal,','setFromAxisAngle(axis,')
        if "type: 'revolute'" in body and 'limits, set(v: number)' in body and not re.search(r'\banchor\b',re.sub(r'//[^\n]*','',body)):
            pivot=re.search(PATTERNS['pivot'],body)
            if not pivot:raise ValueError('Revolute source has no explicit pivot')
            # anchor is unused by the revolute setter; adding it is inert.
            body=body[:pivot.end()]+f'\n    const anchor = new THREE.Vector3({pivot[1]});'+body[pivot.end():]
        return body
    source=re.sub(r'  \{ // part [^\n]+\n.*?\n  \}',normalize_rotated_axis,source,flags=re.S)
    pattern=re.compile(r"joints\['([^']+)'\] = \{ type: 'revolute', part: '([^']+)', limits: \[([^]]+)\], set\(v: number\) \{ v = Math.min\(this.limits\[1\], Math.max\(this.limits\[0\], v\)\); pivot.quaternion.copy\(frame\).multiply\(new THREE.Quaternion\(\).setFromAxisAngle\(new THREE.Vector3\(([^)]+)\)(\.normalize\(\))?, v\)\); \} \};")
    edits=[]
    for block in re.finditer(r'  \{ // part ([^\n]+)\n(.*?)\n  \}',source,re.S):
        body=block[2];match=pattern.search(body)
        if not match:continue
        if any(re.search(r'const '+key+r'\s*[:=]',body) for key in ('axis','limits','sign','anchor')):
            raise ValueError('Inline setter also has potentially dependent parameter declarations')
        anchor=re.search(PATTERNS['pivot'],body)
        if not anchor:raise ValueError('Inline revolute setter has no explicit pivot position')
        joint,part,limits,axis,normalize=match.groups()
        text=(f'const axis = new THREE.Vector3({axis}){normalize or ""};\n    '
              f'const limits: [number, number] = [{limits}];\n    const sign = 1.0;\n    '
              f'const anchor = new THREE.Vector3({anchor[1]});\n    '
              f"joints['{joint}'] = {{ type: 'revolute', part: '{part}', limits, set(v: number) {{ v = Math.min(limits[1], Math.max(limits[0], v)); pivot.quaternion.copy(frame).multiply(new THREE.Quaternion().setFromAxisAngle(axis, sign * v)); }} }};")
        edits.append((block.start(2)+match.start(),block.start(2)+match.end(),text))
    for start,end,text in reversed(edits):source=source[:start]+text+source[end:]
    late_pattern=(r"  \{\s*const _piv = groups\['joint:([^']+)'\];\s*const _g = groups\['([^']+)'\];\s*"
                  r'const _want = new THREE\.Vector3\(([^)]+)\);\s*_piv\.updateMatrixWorld\(true\);\s*'
                  r'const _cur = new THREE\.Vector3\(\);\s*_g\.getWorldPosition\(_cur\);\s*'
                  r'_piv\.worldToLocal\(_want\);\s*_g\.position\.copy\(_want\.clone\(\)\.sub\(_cur\)\.add\(_g\.position\)\);\s*\}')
    late=list(re.finditer(late_pattern,source))
    if len(late)>1:raise ValueError('Multiple late position edits require dependency review')
    if late:
        match=late[0];joint,name,want=match.groups();parts,_=parse(source)
        if parts[name]['joint']!=joint:raise ValueError('Late position edit uses another pivot')
        if re.search(r'root\.(?:position|rotation|quaternion|scale)',source):raise ValueError('Transformed root in late position edit')
        parents={child:parent for parent,child in re.findall(r"groups\['([^']+)'\]\.add\(groups\['([^']+)'\]\);",source)}
        for child in re.findall(r"root\.add\(groups\['([^']+)'\]\);",source):parents[child]='__root__'
        local={'__root__':np.eye(4)}
        for part_name,part in parts.items():
            values=part['values'];transform=np.eye(4);transform[:3,:3]=matrix(values['rotation']);transform[:3,3]=values['position'];local[part_name]=transform
            if part['joint']:
                key='joint:'+part['joint'];parents[part_name]=key
                transform=np.eye(4);transform[:3,:3]=matrix(values['frame']);transform[:3,3]=values['pivot'];local[key]=transform
        def world(node,seen=()):
            if node in seen:raise ValueError('Cycle in part hierarchy')
            if node=='__root__':return local[node]
            return world(parents[node],seen+(node,))@local[node]
        desired=np.r_[[number(x) for x in want.split(',')],1.]
        local_want=(np.linalg.inv(world('joint:'+joint))@desired)[:3]
        current=world(name)[:3,3]
        position=local_want-current+np.array(parts[name]['values']['position'])
        text=match[0];source=patch(source,parts,name,{'position':position}).replace(text,'',1)
    return source
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
            fields=raw.split(',')
            if key in ('frame','rotation') and len(fields)==4:
                order=ast.literal_eval(fields[3].strip())
                if order not in ('XYZ','XZY','YXZ','YZX','ZXY','ZYX'):raise ValueError('Unsupported Euler order')
                angles=[number(x) for x in fields[:3]];m_rotation=np.eye(3)
                for axis_name in order:
                    vector=[0.,0.,0.];vector['XYZ'.index(axis_name)]=angles['XYZ'.index(axis_name)]
                    m_rotation=m_rotation@matrix(vector)
                values[key]=euler(m_rotation)
            else:values[key]=tuple(number(x) for x in fields)
            if key=='axis' and '.normalize()' not in m[0] and abs(np.linalg.norm(values[key])-1)>1e-9:
                raise ValueError('Unnormalized nonunit axis requires Gold review')
            spans[key]=(block.start(2)+m.start(),block.start(2)+m.end())
            stripped=stripped.replace(m[0],'',1)
        if 'axis' in values and 'frame' not in values:
            identity=re.search(r'const frame = new THREE\.Quaternion\(\);',body)
            if identity:
                values['frame']=(0.,0.,0.);spans['frame']=(block.start(2)+identity.start(),block.start(2)+identity.end())
                stripped=stripped.replace(identity[0],'',1)
            m=re.search(r'const frame = new THREE\.Quaternion\(\)\.setFromUnitVectors\(axis, new THREE\.Vector3\((.*?)\)(\.normalize\(\))?\);',body)
            if m:
                target=np.array([number(x) for x in m[1].split(',')]);axis=np.array(values['axis'])
                if target.shape!=(3,):raise ValueError('Gold frame target must have three coordinates')
                if m[2]:target/=np.linalg.norm(target)
                axis/=np.linalg.norm(axis);r=float(axis@target)+1
                if r<np.finfo(float).eps:
                    q=np.array([-axis[1],axis[0],0.,0.]) if abs(axis[0])>abs(axis[2]) else np.array([0.,-axis[2],axis[1],0.])
                else:q=np.r_[np.cross(axis,target),r]
                q/=np.linalg.norm(q);x,y,z,w=q
                mat=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                values['frame']=euler(mat);spans['frame']=(block.start(2)+m.start(),block.start(2)+m.end())
                stripped=stripped.replace(m[0],'',1)
        # A construction-only pivot rotation is overwritten by the subsequent
        # quaternion copy before any object is returned or rendered.
        for dead in re.finditer(r'pivot\.rotation\.set\([^;]+\);',body):
            copy_at=body.find('pivot.quaternion.copy(frame);')
            if dead.end()<copy_at:stripped=stripped.replace(dead[0],'',1)
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
    # A frame expression can depend on axis. Freeze its current value when
    # changing axis privately so the counterfactual isolates the motion axis.
    if 'axis' in updates:updates={**updates,'frame':p['values']['frame']}
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

def normalize_for_reference(source,reference):
    """Re-express joints moved between static parents without changing motion."""
    source=canonicalize(source);reference=canonicalize(reference)
    edge_pattern=r"groups\['([^']+)'\]\.add\(groups\['([^']+)'\]\);"
    edges={child:parent for parent,child in re.findall(edge_pattern,source)}
    refedges={child:parent for parent,child in re.findall(edge_pattern,reference)}
    if edges==refedges:return source
    if edges.keys()!=refedges.keys():raise ValueError('Changed hierarchy nodes require review')
    parts,_=parse(source)
    roots=re.findall(r"root\.add\(groups\['([^']+)'\]\);",source)
    if re.search(r'root\.(?:position|rotation|quaternion|scale)',source):raise ValueError('Transformed hierarchy root')
    def static_world(name,seen=()):
        if name in seen or name not in parts or parts[name]['joint']:
            raise ValueError('Reparenting across a moving ancestor requires review')
        p=parts[name]['values'];local=np.eye(4);local[:3,:3]=matrix(p['rotation']);local[:3,3]=p['position']
        return local if name in roots else static_world(edges[name],seen+(name,))@local
    for child,parent in edges.items():
        target=refedges[child]
        if parent==target:continue
        if not child.startswith('joint:'):raise ValueError('Reparented non-joint part requires review')
        names=[name for name,p in parts.items() if 'joint:'+str(p['joint'])==child]
        if len(names)!=1:raise ValueError('Ambiguous reparented joint')
        name=names[0];transform=np.linalg.inv(static_world(target))@static_world(parent)
        parts,_=parse(source);values=parts[name]['values']
        updates={'frame':euler(transform[:3,:3]@matrix(values['frame'])),
                 'pivot':(transform@np.r_[values['pivot'],1.])[:3],
                 'anchor':(transform@np.r_[values['anchor'],1.])[:3]}
        source=patch(source,parts,name,updates)
        source=source.replace(f"groups['{parent}'].add(groups['{child}']);",f"groups['{target}'].add(groups['{child}']);",1)
    return source

def deviations(source,reference):
    source,reference=normalize_for_reference(source,reference),canonicalize(reference)
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
        anchor_key='anchor' if part['joint_type']=='prismatic' else 'pivot'
        aa,ba=np.array(a[anchor_key]),np.array(b[anchor_key])
        origin=aa+fa@a['position'];reforigin=ba+fb@b['position']
        if different(origin,reforigin):add('structure.part_offset',{'position':fa.T@(reforigin-aa)})
        reference_axis=fb@b['axis'];reference_axis/=np.linalg.norm(reference_axis)
        anchor_changed=(np.linalg.norm(np.cross(aa-ba,reference_axis))>1e-9) if part['joint_type']=='revolute' else different(aa,ba)
        if anchor_changed:
            add('kinematics.joint_anchor',{'pivot':b['pivot'],'anchor':b['anchor'],'position':fa.T@(origin-ba)})
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
