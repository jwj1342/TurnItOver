"use strict";
(() => {
  const cases=JSON.parse(document.getElementById('case-data').textContent);
  const packageId=document.documentElement.dataset.package, states=['rest','mid','limit','mid'];
  const owner=document.getElementById('reviewer'), progress=document.getElementById('progress');
  const timers=new Set(); let records=new Map(), warning='';
  const blank=c=>({case_id:c.case_id,reviewer:'',initial_verdict:null,initial_notes:'',initial_recorded_at:null,
    final_decision:null,final_notes:'',supplementary_decisive:null,wrong_part_description:'',reviewed_at:null,
    observed_views:[c.initial_view],view_switches:[],review_status:'pending'});
  const identified=()=>owner.value.trim();
  const key=()=>`hard-pilot-review:${packageId}:${identified()}`;
  const complete=r=>r.initial_recorded_at!==null && r.final_decision!==null && r.final_notes.trim()!=='' &&
    (!['hard','medium_or_easy'].includes(r.final_decision)||r.wrong_part_description.trim()!=='') &&
    (r.observed_views.length<2 || typeof r.supplementary_decisive==='boolean');
  const summarize=()=>{progress.textContent=`${[...records.values()].filter(complete).length} / ${cases.length} 已填写（待交回，不代表通过）。${warning}`;};
  const save=()=>{if(identified()){try{localStorage.setItem(key(),JSON.stringify([...records.values()]));}catch{warning='本地保存不可用，请及时导出。';}}summarize();};
  const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text)e.textContent=text;if(cls)e.className=cls;return e;};
  const control=(parent,title,node)=>{const label=el('label',title);label.append(node);parent.append(label);return node;};
  const select=(parent,title,values,current)=>{const s=el('select');s.append(new Option('请选择',''));values.forEach(([v,t])=>s.append(new Option(t,v)));s.value=current??'';return control(parent,title,s);};
  const notes=(parent,title,current)=>{const n=el('textarea');n.value=current;return control(parent,title,n);};
  const render=()=>{
    timers.forEach(clearInterval);timers.clear();document.getElementById('cases').replaceChildren();
    cases.forEach(c=>{
      const r=records.get(c.case_id), card=el('section');card.dataset.id=c.case_id;card.append(el('h2',c.case_id));
      let index=0, view=c.initial_view, timer=null;
      const motion=el('div',null,'motion'),images={};
      ['reference','candidate'].forEach(role=>{const f=el('figure');f.append(el('figcaption',role==='reference'?'参考':'候选'));images[role]=el('img');images[role].width=images[role].height=256;images[role].alt=`${role} 原生图像`;f.append(images[role]);motion.append(f);});card.append(motion);
      const state=el('p',null,'state');card.append(state);
      const show=()=>{const s=states[index];Object.entries(images).forEach(([role,img])=>img.src=`${c.native_root}/${view}/${role}_${s}.png`);state.textContent=`${view===c.initial_view?'初始':'补充'}视角 · ${s}`;};show();
      const step=el('button','下一状态'),play=el('button','播放三态');step.className='step';play.className='play';card.append(step,play);
      step.onclick=()=>{index=(index+1)%4;show();};play.onclick=()=>{if(timer){clearInterval(timer);timers.delete(timer);timer=null;play.textContent='播放三态';}else{timer=setInterval(()=>{index=(index+1)%4;show();},700);timers.add(timer);play.textContent='暂停';}};
      const first=el('fieldset');first.append(el('legend','第一阶段：初始视角'));card.append(first);
      const initial=select(first,'初始判断',[['clean','未发现异常'],['uncertain','当前证据不足'],['defect','可以识别异常']],r.initial_verdict);
      const firstNotes=notes(first,'初始依据（记录后锁定）',r.initial_notes),lock=el('button','记录初始判断');lock.className='record-initial';first.append(lock);
      const switchView=el('button','打开唯一补充视角');switchView.className='switch-view';card.append(switchView);
      const final=el('fieldset');final.append(el('legend','最终可观察性与难度'));card.append(final);
      const decision=select(final,'最终判断',[['hard','符合 Hard'],['medium_or_easy','较明显，更接近 Medium/Easy'],['too_subtle','仍难以辨认，证据不足'],['clean','未发现缺陷'],['invalid','出框、遮挡或其他无效情况']],r.final_decision);
      const decisive=select(final,'补充视角是否给出决定性证据',[['yes','是'],['no','否']],r.supplementary_decisive===null?'':r.supplementary_decisive?'yes':'no');
      const description=notes(final,'错误部件或关节的自由描述（Hard/较明显时必填）',r.wrong_part_description);
      const finalNotes=notes(final,'最终判断依据（必填）',r.final_notes),status=el('p',null,'status');card.append(status);
      const refresh=()=>{const locked=r.initial_recorded_at!==null;first.disabled=!identified()||locked;lock.disabled=!identified()||locked||!initial.value||!firstNotes.value.trim();switchView.disabled=!locked;final.disabled=!identified()||!locked;decisive.disabled=r.observed_views.length<2;status.textContent=complete(r)?'已填写，待单独交回；不代表样本通过':'待填写';};
      const update=()=>{r.reviewer=identified();r.final_decision=decision.value||null;r.final_notes=finalNotes.value;r.wrong_part_description=description.value;r.supplementary_decisive=decisive.value===''?null:decisive.value==='yes';r.review_status=complete(r)?'completed':'pending';r.reviewed_at=complete(r)?new Date().toISOString():null;save();refresh();};
      initial.onchange=()=>{r.initial_verdict=initial.value||null;save();refresh();};firstNotes.oninput=()=>{r.initial_notes=firstNotes.value;save();refresh();};
      lock.onclick=()=>{if(!initial.value||!firstNotes.value.trim()||!identified())return;r.reviewer=identified();r.initial_verdict=initial.value;r.initial_notes=firstNotes.value;r.initial_recorded_at=new Date().toISOString();save();refresh();};
      switchView.onclick=()=>{if(!r.initial_recorded_at)return;const next=view===c.initial_view?c.supplementary_view:c.initial_view;r.view_switches.push({from:view,to:next,state:states[index],at:new Date().toISOString()});view=next;if(!r.observed_views.includes(view))r.observed_views.push(view);switchView.textContent=view===c.initial_view?'回看补充视角':'回看初始视角';show();r.review_status=complete(r)?'completed':'pending';if(!complete(r))r.reviewed_at=null;save();refresh();};
      [decision,decisive].forEach(n=>n.onchange=update);[description,finalNotes].forEach(n=>n.oninput=update);refresh();document.getElementById('cases').append(card);
    });summarize();
  };
  const reset=()=>{warning='';records=new Map(cases.map(c=>[c.case_id,blank(c)]));render();};
  owner.onchange=reset;
  document.getElementById('resume').onclick=()=>{if(!identified()){warning='请先填写审核者标识。';summarize();return;}try{const rows=JSON.parse(localStorage.getItem(key())||'null');if(!Array.isArray(rows)||rows.length!==cases.length||new Set(rows.map(r=>r.case_id)).size!==cases.length||rows.some(r=>r.reviewer!==identified()&&!(!r.reviewer&&r.review_status==='pending'))||rows.some(r=>!cases.some(c=>c.case_id===r.case_id)))throw Error();records=new Map(rows.map(r=>[r.case_id,r]));warning='';render();}catch{warning='没有匹配的本人记录；保持当前填写。';summarize();}};
  document.getElementById('export').onclick=()=>{const rows=cases.map(c=>({...records.get(c.case_id),reviewer:identified()}));const blob=new Blob([JSON.stringify(rows,null,2)],{type:'application/json'});const a=el('a');a.href=URL.createObjectURL(blob);a.download='hard-pilot-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
  reset();
})();
