"use strict";
(() => {
  const cases = JSON.parse(document.getElementById("case-data").textContent);
  const key = "medium-review-v1:" + cases.map(c => c.case_id).join(":") + ":";
  const states = ["rest", "mid", "limit", "mid"];
  const decisions = ["medium", "clean", "too_obvious", "too_subtle", "invalid"];
  const reviewer = document.getElementById("reviewer"), progress = document.getElementById("progress");
  const blank = c => ({case_id:c.case_id, reviewer:"", decision:null, notes:"", reviewed_at:null});
  const el = (tag, text, cls) => {const node=document.createElement(tag);if(text)node.textContent=text;if(cls)node.className=cls;return node;};
  let rows=cases.map(blank), activeReviewer="", warning="", timers=[];
  const persist = () => {
    if(activeReviewer)try {localStorage.setItem(key+activeReviewer,JSON.stringify(rows));}
    catch {warning="本地存储不可用，请及时导出备份。";}
    progress.textContent=`${rows.filter(r=>r.decision!==null).length} / ${cases.length} 已填写。${warning}`;
  };
  const render = () => {
    timers.forEach(clearInterval);timers=[];document.getElementById("cases").replaceChildren();
    cases.forEach((c,i) => {
      const row=rows[i],card=el("section");card.dataset.id=c.case_id;card.append(el("h2",`${i+1}. ${c.case_id}`));
      const motion=el("div",null,"motion"),images={};
      for(const role of ["reference","candidate"]){const figure=el("figure"),img=el("img");img.src=c[role].rest;img.alt=role;img.className=role;images[role]=img;figure.append(el("figcaption",role==="reference"?"参考":"候选"),img);motion.append(figure);}
      const state=el("p","rest","state"),step=el("button","下一状态","step"),play=el("button","播放三态","play");
      let index=0,timer=null;const show=()=>{state.textContent=states[index];for(const role of ["reference","candidate"])images[role].src=c[role][states[index]];};
      step.onclick=()=>{index=(index+1)%states.length;show();};play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent="播放三态";}else{timer=setInterval(()=>{index=(index+1)%states.length;show();},700);timers.push(timer);play.textContent="暂停";}};
      const details=el("details"),board=el("div",null,"board");details.append(el("summary","三态并排比较"));
      for(const role of ["reference","candidate"])for(const s of ["rest","mid","limit"]){const cell=el("div"),img=el("img");img.src=c[role][s];img.alt=`${role} ${s}`;cell.append(el("p",`${role==="reference"?"参考":"候选"} ${s}`),img);board.append(cell);}details.append(board);
      const label=el("label","判断 "),select=el("select");
      for(const [value,text] of [["","请选择"],["medium","仔细比较或思考可确认问题"],["clean","未发现问题／正常"],["too_obvious","问题过于明显"],["too_subtle","差异过于隐蔽"],["invalid","无法判断或显示有问题"]]){const option=el("option",text);option.value=value;select.append(option);}select.value=row.decision||"";select.disabled=!activeReviewer;
      const notes=el("textarea");notes.placeholder="请说明部件、状态及观察依据";notes.value=row.notes;notes.disabled=!activeReviewer;
      const save=()=>{row.decision=select.value||null;row.notes=notes.value;row.reviewer=activeReviewer;row.reviewed_at=row.decision?new Date().toISOString():null;persist();};
      select.onchange=save;notes.oninput=save;label.append(select);card.append(motion,state,step,play,details,label,notes);document.getElementById("cases").append(card);
    });
    persist();
  };
  document.getElementById("start").onclick=()=>{
    const name=reviewer.value.trim();if(!name || name.length>128){alert("请填写本人审核者标识（1–128个字符）。");return;}
    activeReviewer=name;rows=cases.map(blank);warning="";
    try {
      const saved=JSON.parse(localStorage.getItem(key+name)||"null");
      if(saved){const keys=Object.keys(blank(cases[0])).sort().join();
        if(!Array.isArray(saved)||saved.length!==cases.length||!saved.every((r,i)=>r.case_id===cases[i].case_id&&r.reviewer===name&&Object.keys(r).sort().join()===keys&&typeof r.notes==="string"&&(r.decision===null?r.reviewed_at===null:decisions.includes(r.decision)&&typeof r.reviewed_at==="string"&&!Number.isNaN(Date.parse(r.reviewed_at)))))throw Error("invalid saved response");
        rows=saved;
      }
    } catch {warning="本人本地记录不可用，现显示空白；请保留已导出的备份。";}
    rows.forEach(r=>{r.reviewer=name;});render();
  };
  reviewer.oninput=()=>{if(activeReviewer&&reviewer.value.trim()!==activeReviewer){activeReviewer="";rows=cases.map(blank);warning="请点击开始／载入本人审核。";render();}};
  document.getElementById("export").onclick=()=>{
    if(rows.some(r=>r.decision!==null)&&!activeReviewer){alert("请先载入本人审核。");return;}
    const exported=rows.map(r=>({...r,reviewer:activeReviewer}));
    const url=URL.createObjectURL(new Blob([exported.map(r=>JSON.stringify(r)).join("\n")+"\n"],{type:"application/x-ndjson;charset=utf-8"}));
    const a=el("a");a.href=url;a.download="medium-review-return.jsonl";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  render();
})();
